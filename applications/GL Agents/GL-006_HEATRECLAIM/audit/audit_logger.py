"""
GL-006 HEATRECLAIM - Audit Logger

Comprehensive audit logging system for regulatory compliance.
Implements structured logging with 6 event types, SHA-256 integrity
verification, and correlation tracking for full traceability.

Event Types:
1. DESIGN_SUBMISSION - Heat exchanger design inputs and configurations
2. OPTIMIZATION_RUN - MILP optimization executions and results
3. SAFETY_CHECK - Safety constraint validations and violations
4. CONFIG_CHANGE - System configuration modifications
5. USER_ACTION - User interactions and decisions
6. SYSTEM_EVENT - System lifecycle and operational events

Standards:
- ISO 27001: Information security management
- SOC 2 Type II: Service organization controls
- 21 CFR Part 11: Electronic records and signatures (FDA)
"""

import hashlib
import json
import logging
import os
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    """Six audit event types for comprehensive coverage."""

    DESIGN_SUBMISSION = "design_submission"
    OPTIMIZATION_RUN = "optimization_run"
    SAFETY_CHECK = "safety_check"
    CONFIG_CHANGE = "config_change"
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"


class AuditSeverity(str, Enum):
    """Severity levels for audit events."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditOutcome(str, Enum):
    """Outcome of audited operation."""

    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class AuditContext:
    """Immutable context for audit events."""

    correlation_id: str
    session_id: str
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    request_id: Optional[str] = None


@dataclass(frozen=True)
class AuditEvent:
    """Immutable audit event record."""

    event_type: AuditEventType
    action: str
    severity: AuditSeverity = AuditSeverity.INFO
    outcome: AuditOutcome = AuditOutcome.SUCCESS
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditRecord:
    """Complete audit record with provenance."""

    record_id: str
    timestamp: str
    event_type: str
    action: str
    severity: str
    outcome: str
    message: str
    details: Dict[str, Any]
    metadata: Dict[str, Any]
    context: Dict[str, Any]
    provenance: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditRecord":
        """Create from dictionary."""
        return cls(**data)


class AuditLogger:
    """
    Enterprise-grade audit logger for GL-006 HEATRECLAIM.

    Provides comprehensive audit logging with:
    - 6 event types covering all operational aspects
    - SHA-256 hash chain for tamper detection
    - Correlation IDs for request tracing
    - Multiple storage backends
    - Async batch processing for performance

    Usage:
        audit = AuditLogger()

        # Log a design submission
        audit.log_design_submission(
            design_id="HEX-001",
            hot_streams=5,
            cold_streams=4,
            user_id="engineer@example.com"
        )

        # Log safety check with context manager
        with audit.audit_safety_check("HEX-001") as ctx:
            result = validator.validate(design)
            ctx.set_outcome("success" if result.is_safe else "failure")
    """

    _instance: Optional["AuditLogger"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs) -> "AuditLogger":
        """Singleton pattern for global audit logger."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        storage: Optional["AuditStorage"] = None,
        app_name: str = "GL-006-HEATRECLAIM",
        environment: str = "production",
        enable_hash_chain: bool = True,
    ):
        """
        Initialize audit logger.

        Args:
            storage: Audit storage backend
            app_name: Application identifier
            environment: Deployment environment
            enable_hash_chain: Enable SHA-256 hash chain for integrity
        """
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.app_name = app_name
        self.environment = environment
        self.enable_hash_chain = enable_hash_chain
        self._storage = storage
        self._previous_hash: Optional[str] = None
        self._record_count = 0
        self._local = threading.local()
        self._initialized = True

        logger.info(
            f"AuditLogger initialized: app={app_name}, env={environment}, "
            f"hash_chain={enable_hash_chain}"
        )

    def set_storage(self, storage: "AuditStorage") -> None:
        """Set storage backend."""
        self._storage = storage

    def _get_context(self) -> AuditContext:
        """Get current audit context from thread-local storage."""
        return getattr(
            self._local,
            "context",
            AuditContext(
                correlation_id=str(uuid.uuid4()),
                session_id=str(uuid.uuid4()),
            )
        )

    def set_context(self, context: AuditContext) -> None:
        """Set audit context for current thread."""
        self._local.context = context

    @contextmanager
    def correlation_context(
        self,
        correlation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs,
    ):
        """
        Context manager for correlation tracking.

        Args:
            correlation_id: Optional correlation ID (generated if not provided)
            user_id: Optional user identifier
            **kwargs: Additional context fields
        """
        old_context = getattr(self._local, "context", None)

        new_context = AuditContext(
            correlation_id=correlation_id or str(uuid.uuid4()),
            session_id=str(uuid.uuid4()),
            user_id=user_id,
            **kwargs,
        )
        self._local.context = new_context

        try:
            yield new_context
        finally:
            if old_context:
                self._local.context = old_context
            else:
                delattr(self._local, "context")

    def _compute_hash(self, record: AuditRecord) -> str:
        """Compute SHA-256 hash for audit record."""
        content = record.to_json()
        if self._previous_hash:
            content = self._previous_hash + content
        return hashlib.sha256(content.encode()).hexdigest()

    def _create_record(self, event: AuditEvent) -> AuditRecord:
        """Create complete audit record from event."""
        context = self._get_context()

        record = AuditRecord(
            record_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event.event_type.value,
            action=event.action,
            severity=event.severity.value,
            outcome=event.outcome.value,
            message=event.message,
            details=dict(event.details),
            metadata={
                "app_name": self.app_name,
                "environment": self.environment,
                "record_sequence": self._record_count,
                **event.metadata,
            },
            context={
                "correlation_id": context.correlation_id,
                "session_id": context.session_id,
                "user_id": context.user_id,
                "tenant_id": context.tenant_id,
                "source_ip": context.source_ip,
                "request_id": context.request_id,
            },
            provenance={},
        )

        # Compute hash chain
        if self.enable_hash_chain:
            record_hash = self._compute_hash(record)
            record.provenance["record_hash"] = record_hash
            record.provenance["previous_hash"] = self._previous_hash or "genesis"
            record.provenance["hash_algorithm"] = "SHA-256"
            self._previous_hash = record_hash

        self._record_count += 1
        return record

    def log(self, event: AuditEvent) -> AuditRecord:
        """
        Log an audit event.

        Args:
            event: Audit event to log

        Returns:
            Complete audit record
        """
        record = self._create_record(event)

        # Persist to storage if configured
        if self._storage:
            self._storage.store(record)

        # Also log to standard logger
        log_method = getattr(logger, event.severity.value, logger.info)
        log_method(
            f"[AUDIT] {event.event_type.value}:{event.action} - {event.message} "
            f"(correlation_id={record.context['correlation_id']})"
        )

        return record

    # =========================================================================
    # EVENT TYPE 1: DESIGN SUBMISSION
    # =========================================================================

    def log_design_submission(
        self,
        design_id: str,
        hot_streams: int,
        cold_streams: int,
        total_duty_kw: float,
        user_id: Optional[str] = None,
        **details,
    ) -> AuditRecord:
        """
        Log a heat exchanger design submission.

        Args:
            design_id: Unique design identifier
            hot_streams: Number of hot streams
            cold_streams: Number of cold streams
            total_duty_kw: Total heat duty in kW
            user_id: User who submitted the design
            **details: Additional design details
        """
        return self.log(AuditEvent(
            event_type=AuditEventType.DESIGN_SUBMISSION,
            action="submit_design",
            severity=AuditSeverity.INFO,
            outcome=AuditOutcome.SUCCESS,
            message=f"Design {design_id} submitted with {hot_streams} hot and {cold_streams} cold streams",
            details={
                "design_id": design_id,
                "hot_streams": hot_streams,
                "cold_streams": cold_streams,
                "total_duty_kw": total_duty_kw,
                "submitted_by": user_id,
                **details,
            },
        ))

    def log_design_validation(
        self,
        design_id: str,
        is_valid: bool,
        validation_errors: Optional[List[str]] = None,
    ) -> AuditRecord:
        """Log design validation result."""
        return self.log(AuditEvent(
            event_type=AuditEventType.DESIGN_SUBMISSION,
            action="validate_design",
            severity=AuditSeverity.INFO if is_valid else AuditSeverity.WARNING,
            outcome=AuditOutcome.SUCCESS if is_valid else AuditOutcome.FAILURE,
            message=f"Design {design_id} validation: {'passed' if is_valid else 'failed'}",
            details={
                "design_id": design_id,
                "is_valid": is_valid,
                "validation_errors": validation_errors or [],
            },
        ))

    # =========================================================================
    # EVENT TYPE 2: OPTIMIZATION RUN
    # =========================================================================

    def log_optimization_start(
        self,
        optimization_id: str,
        algorithm: str,
        objective: str,
        constraints: Dict[str, Any],
    ) -> AuditRecord:
        """Log optimization run start."""
        return self.log(AuditEvent(
            event_type=AuditEventType.OPTIMIZATION_RUN,
            action="start_optimization",
            severity=AuditSeverity.INFO,
            outcome=AuditOutcome.SUCCESS,
            message=f"Optimization {optimization_id} started with {algorithm}",
            details={
                "optimization_id": optimization_id,
                "algorithm": algorithm,
                "objective": objective,
                "constraints": constraints,
            },
        ))

    def log_optimization_complete(
        self,
        optimization_id: str,
        success: bool,
        objective_value: Optional[float] = None,
        iterations: Optional[int] = None,
        solve_time_seconds: Optional[float] = None,
        solution_summary: Optional[Dict[str, Any]] = None,
    ) -> AuditRecord:
        """Log optimization run completion."""
        return self.log(AuditEvent(
            event_type=AuditEventType.OPTIMIZATION_RUN,
            action="complete_optimization",
            severity=AuditSeverity.INFO if success else AuditSeverity.ERROR,
            outcome=AuditOutcome.SUCCESS if success else AuditOutcome.FAILURE,
            message=f"Optimization {optimization_id} {'completed' if success else 'failed'}",
            details={
                "optimization_id": optimization_id,
                "success": success,
                "objective_value": objective_value,
                "iterations": iterations,
                "solve_time_seconds": solve_time_seconds,
                "solution_summary": solution_summary or {},
            },
        ))

    def log_pareto_generation(
        self,
        optimization_id: str,
        num_solutions: int,
        pareto_front: List[Dict[str, float]],
    ) -> AuditRecord:
        """Log Pareto front generation."""
        return self.log(AuditEvent(
            event_type=AuditEventType.OPTIMIZATION_RUN,
            action="generate_pareto",
            severity=AuditSeverity.INFO,
            outcome=AuditOutcome.SUCCESS,
            message=f"Generated Pareto front with {num_solutions} solutions",
            details={
                "optimization_id": optimization_id,
                "num_solutions": num_solutions,
                "pareto_front": pareto_front,
            },
        ))

    # =========================================================================
    # EVENT TYPE 3: SAFETY CHECK
    # =========================================================================

    def log_safety_check(
        self,
        design_id: str,
        check_type: str,
        passed: bool,
        violations: Optional[List[Dict[str, Any]]] = None,
        standard_reference: Optional[str] = None,
    ) -> AuditRecord:
        """
        Log safety constraint check.

        Args:
            design_id: Design being checked
            check_type: Type of safety check
            passed: Whether check passed
            violations: List of violations if any
            standard_reference: Reference standard (e.g., ASME PTC 4.3)
        """
        severity = AuditSeverity.INFO if passed else AuditSeverity.WARNING
        if violations and any(v.get("severity") == "critical" for v in violations):
            severity = AuditSeverity.CRITICAL

        return self.log(AuditEvent(
            event_type=AuditEventType.SAFETY_CHECK,
            action=f"check_{check_type}",
            severity=severity,
            outcome=AuditOutcome.SUCCESS if passed else AuditOutcome.FAILURE,
            message=f"Safety check '{check_type}' for {design_id}: {'PASS' if passed else 'FAIL'}",
            details={
                "design_id": design_id,
                "check_type": check_type,
                "passed": passed,
                "violations": violations or [],
                "standard_reference": standard_reference,
            },
        ))

    def log_safety_override(
        self,
        design_id: str,
        constraint: str,
        original_value: Any,
        override_value: Any,
        justification: str,
        approved_by: str,
    ) -> AuditRecord:
        """Log safety constraint override (requires approval)."""
        return self.log(AuditEvent(
            event_type=AuditEventType.SAFETY_CHECK,
            action="override_constraint",
            severity=AuditSeverity.WARNING,
            outcome=AuditOutcome.SUCCESS,
            message=f"Safety override for {constraint} on {design_id} approved by {approved_by}",
            details={
                "design_id": design_id,
                "constraint": constraint,
                "original_value": original_value,
                "override_value": override_value,
                "justification": justification,
                "approved_by": approved_by,
            },
        ))

    @contextmanager
    def audit_safety_check(self, design_id: str, check_type: str = "full"):
        """
        Context manager for safety check auditing.

        Usage:
            with audit.audit_safety_check("HEX-001") as ctx:
                result = validator.validate(design)
                ctx.add_violation({"type": "pressure_drop", "value": 0.6})
                ctx.set_passed(result.is_safe)
        """
        class SafetyCheckContext:
            def __init__(self):
                self.violations = []
                self.passed = True
                self.standard_reference = None

            def add_violation(self, violation: Dict[str, Any]):
                self.violations.append(violation)
                self.passed = False

            def set_passed(self, passed: bool):
                self.passed = passed

            def set_standard(self, reference: str):
                self.standard_reference = reference

        ctx = SafetyCheckContext()
        try:
            yield ctx
        finally:
            self.log_safety_check(
                design_id=design_id,
                check_type=check_type,
                passed=ctx.passed,
                violations=ctx.violations,
                standard_reference=ctx.standard_reference,
            )

    # =========================================================================
    # EVENT TYPE 4: CONFIG CHANGE
    # =========================================================================

    def log_config_change(
        self,
        config_key: str,
        old_value: Any,
        new_value: Any,
        changed_by: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> AuditRecord:
        """Log configuration change."""
        return self.log(AuditEvent(
            event_type=AuditEventType.CONFIG_CHANGE,
            action="update_config",
            severity=AuditSeverity.INFO,
            outcome=AuditOutcome.SUCCESS,
            message=f"Configuration '{config_key}' changed from {old_value} to {new_value}",
            details={
                "config_key": config_key,
                "old_value": old_value,
                "new_value": new_value,
                "changed_by": changed_by,
                "reason": reason,
            },
        ))

    def log_threshold_update(
        self,
        threshold_name: str,
        old_value: float,
        new_value: float,
        unit: str,
        standard_reference: Optional[str] = None,
    ) -> AuditRecord:
        """Log safety threshold update."""
        return self.log(AuditEvent(
            event_type=AuditEventType.CONFIG_CHANGE,
            action="update_threshold",
            severity=AuditSeverity.WARNING,
            outcome=AuditOutcome.SUCCESS,
            message=f"Threshold '{threshold_name}' updated: {old_value} -> {new_value} {unit}",
            details={
                "threshold_name": threshold_name,
                "old_value": old_value,
                "new_value": new_value,
                "unit": unit,
                "standard_reference": standard_reference,
            },
        ))

    # =========================================================================
    # EVENT TYPE 5: USER ACTION
    # =========================================================================

    def log_user_login(
        self,
        user_id: str,
        success: bool,
        auth_method: str = "password",
        source_ip: Optional[str] = None,
    ) -> AuditRecord:
        """Log user login attempt."""
        return self.log(AuditEvent(
            event_type=AuditEventType.USER_ACTION,
            action="login",
            severity=AuditSeverity.INFO if success else AuditSeverity.WARNING,
            outcome=AuditOutcome.SUCCESS if success else AuditOutcome.FAILURE,
            message=f"User {user_id} login {'successful' if success else 'failed'}",
            details={
                "user_id": user_id,
                "auth_method": auth_method,
                "source_ip": source_ip,
            },
        ))

    def log_user_action(
        self,
        user_id: str,
        action: str,
        resource_type: str,
        resource_id: str,
        success: bool = True,
        **details,
    ) -> AuditRecord:
        """Log generic user action."""
        return self.log(AuditEvent(
            event_type=AuditEventType.USER_ACTION,
            action=action,
            severity=AuditSeverity.INFO,
            outcome=AuditOutcome.SUCCESS if success else AuditOutcome.FAILURE,
            message=f"User {user_id} performed {action} on {resource_type}/{resource_id}",
            details={
                "user_id": user_id,
                "resource_type": resource_type,
                "resource_id": resource_id,
                **details,
            },
        ))

    def log_approval(
        self,
        approver_id: str,
        item_type: str,
        item_id: str,
        approved: bool,
        comments: Optional[str] = None,
    ) -> AuditRecord:
        """Log approval decision."""
        return self.log(AuditEvent(
            event_type=AuditEventType.USER_ACTION,
            action="approve" if approved else "reject",
            severity=AuditSeverity.INFO,
            outcome=AuditOutcome.SUCCESS,
            message=f"{item_type} {item_id} {'approved' if approved else 'rejected'} by {approver_id}",
            details={
                "approver_id": approver_id,
                "item_type": item_type,
                "item_id": item_id,
                "approved": approved,
                "comments": comments,
            },
        ))

    # =========================================================================
    # EVENT TYPE 6: SYSTEM EVENT
    # =========================================================================

    def log_system_startup(
        self,
        version: str,
        config_hash: Optional[str] = None,
    ) -> AuditRecord:
        """Log system startup."""
        return self.log(AuditEvent(
            event_type=AuditEventType.SYSTEM_EVENT,
            action="startup",
            severity=AuditSeverity.INFO,
            outcome=AuditOutcome.SUCCESS,
            message=f"System started, version {version}",
            details={
                "version": version,
                "config_hash": config_hash,
            },
        ))

    def log_system_shutdown(
        self,
        reason: str = "normal",
        uptime_seconds: Optional[float] = None,
    ) -> AuditRecord:
        """Log system shutdown."""
        return self.log(AuditEvent(
            event_type=AuditEventType.SYSTEM_EVENT,
            action="shutdown",
            severity=AuditSeverity.INFO,
            outcome=AuditOutcome.SUCCESS,
            message=f"System shutdown: {reason}",
            details={
                "reason": reason,
                "uptime_seconds": uptime_seconds,
            },
        ))

    def log_system_error(
        self,
        error_type: str,
        error_message: str,
        component: str,
        stack_trace: Optional[str] = None,
    ) -> AuditRecord:
        """Log system error."""
        return self.log(AuditEvent(
            event_type=AuditEventType.SYSTEM_EVENT,
            action="error",
            severity=AuditSeverity.ERROR,
            outcome=AuditOutcome.FAILURE,
            message=f"System error in {component}: {error_type}",
            details={
                "error_type": error_type,
                "error_message": error_message,
                "component": component,
                "stack_trace": stack_trace,
            },
        ))

    def log_health_check(
        self,
        status: str,
        components: Dict[str, str],
        latency_ms: Optional[float] = None,
    ) -> AuditRecord:
        """Log health check result."""
        return self.log(AuditEvent(
            event_type=AuditEventType.SYSTEM_EVENT,
            action="health_check",
            severity=AuditSeverity.DEBUG,
            outcome=AuditOutcome.SUCCESS if status == "healthy" else AuditOutcome.PARTIAL,
            message=f"Health check: {status}",
            details={
                "status": status,
                "components": components,
                "latency_ms": latency_ms,
            },
        ))

    # =========================================================================
    # QUERY INTERFACE
    # =========================================================================

    def get_records(
        self,
        event_type: Optional[AuditEventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        correlation_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[AuditRecord]:
        """
        Query audit records.

        Args:
            event_type: Filter by event type
            start_time: Filter by start time
            end_time: Filter by end time
            correlation_id: Filter by correlation ID
            limit: Maximum records to return

        Returns:
            List of matching audit records
        """
        if self._storage is None:
            logger.warning("No storage configured, cannot query records")
            return []

        from .audit_storage import AuditQuery

        query = AuditQuery(
            event_type=event_type.value if event_type else None,
            start_time=start_time,
            end_time=end_time,
            correlation_id=correlation_id,
            limit=limit,
        )

        result = self._storage.query(query)
        return result.records


# Type alias for storage interface
AuditStorage = Any  # Defined in audit_storage.py


# =============================================================================
# CALCULATION EVENT LOGGING
# =============================================================================

@dataclass
class CalculationAuditEvent:
    """
    Comprehensive calculation event for audit trail.

    Captures all aspects of a calculation for regulatory compliance:
    - Input data with SHA-256 hash
    - Output data with SHA-256 hash
    - Formula/method identification
    - Timing and performance metrics
    - Provenance chain for reproducibility
    """

    calculation_id: str
    calculation_type: str
    timestamp: str

    # Input provenance
    input_summary: Dict[str, Any]
    input_hash: str
    input_count: int

    # Output provenance
    output_summary: Dict[str, Any]
    output_hash: str

    # Formula/method tracking
    formula_id: str
    formula_version: str
    algorithm: str

    # Computation metadata
    is_deterministic: bool
    random_seed: Optional[int]
    solver_used: Optional[str]
    iterations: Optional[int]

    # Performance
    calculation_time_ms: float
    memory_usage_mb: Optional[float]

    # Status
    success: bool
    error_message: Optional[str]
    warnings: List[str]

    # Provenance chain
    parent_calculation_id: Optional[str]
    provenance_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "calculation_id": self.calculation_id,
            "calculation_type": self.calculation_type,
            "timestamp": self.timestamp,
            "input_summary": self.input_summary,
            "input_hash": self.input_hash,
            "input_count": self.input_count,
            "output_summary": self.output_summary,
            "output_hash": self.output_hash,
            "formula_id": self.formula_id,
            "formula_version": self.formula_version,
            "algorithm": self.algorithm,
            "is_deterministic": self.is_deterministic,
            "random_seed": self.random_seed,
            "solver_used": self.solver_used,
            "iterations": self.iterations,
            "calculation_time_ms": self.calculation_time_ms,
            "memory_usage_mb": self.memory_usage_mb,
            "success": self.success,
            "error_message": self.error_message,
            "warnings": self.warnings,
            "parent_calculation_id": self.parent_calculation_id,
            "provenance_hash": self.provenance_hash,
        }


class CalculationEventLogger:
    """
    Specialized logger for calculation events with full provenance tracking.

    Provides comprehensive audit logging for all calculations performed by
    GL-006 HEATRECLAIM with SHA-256 integrity verification and
    regulatory compliance support.

    Supports all calculation types:
    - Pinch analysis
    - LMTD calculations
    - Exergy analysis
    - Economic analysis
    - HEN synthesis
    - MILP optimization
    - Pareto generation
    - Uncertainty quantification

    Example:
        >>> calc_logger = CalculationEventLogger(audit_logger)
        >>> with calc_logger.track_calculation("pinch_analysis") as tracker:
        ...     tracker.set_inputs({"streams": 10, "delta_t_min": 10})
        ...     result = perform_pinch_analysis(streams)
        ...     tracker.set_outputs({"pinch_temp": 85.5, "min_utility": 1500})
        >>> # Calculation automatically logged with full provenance
    """

    VERSION = "1.0.0"

    # Standard calculation types
    CALCULATION_TYPES = {
        "pinch_analysis": "PINCH",
        "lmtd_calculation": "LMTD",
        "ntu_calculation": "NTU",
        "exergy_analysis": "EXERGY",
        "economic_analysis": "ECON",
        "hen_synthesis": "HEN",
        "milp_optimization": "MILP",
        "pareto_generation": "PARETO",
        "uncertainty_quantification": "UQ",
        "safety_validation": "SAFETY",
        "shap_explanation": "SHAP",
        "sensitivity_analysis": "SENS",
    }

    def __init__(
        self,
        audit_logger: AuditLogger,
        enable_performance_tracking: bool = True,
        enable_memory_tracking: bool = False,
    ):
        """
        Initialize calculation event logger.

        Args:
            audit_logger: Parent audit logger for persistence
            enable_performance_tracking: Track calculation timing
            enable_memory_tracking: Track memory usage (slight overhead)
        """
        self.audit_logger = audit_logger
        self.enable_performance_tracking = enable_performance_tracking
        self.enable_memory_tracking = enable_memory_tracking
        self._calculation_count = 0
        self._lock = threading.Lock()

        logger.info(
            f"CalculationEventLogger initialized: "
            f"performance={enable_performance_tracking}, "
            f"memory={enable_memory_tracking}"
        )

    def _compute_hash(self, data: Any) -> str:
        """Compute SHA-256 hash of data."""
        if isinstance(data, dict):
            json_str = json.dumps(data, sort_keys=True, default=str)
        else:
            json_str = str(data)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB."""
        if not self.enable_memory_tracking:
            return None
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return None
        except Exception:
            return None

    @contextmanager
    def track_calculation(
        self,
        calculation_type: str,
        formula_id: Optional[str] = None,
        algorithm: Optional[str] = None,
        parent_id: Optional[str] = None,
        random_seed: Optional[int] = None,
    ):
        """
        Context manager for tracking a calculation with full provenance.

        Args:
            calculation_type: Type of calculation (e.g., "pinch_analysis")
            formula_id: Identifier for the formula/method used
            algorithm: Algorithm name
            parent_id: Parent calculation ID for chained calculations
            random_seed: Random seed if calculation uses randomness

        Yields:
            CalculationTracker object for setting inputs/outputs

        Example:
            >>> with calc_logger.track_calculation("exergy_analysis") as tracker:
            ...     tracker.set_inputs(input_data)
            ...     result = calculate_exergy(data)
            ...     tracker.set_outputs(result)
            ...     if warning:
            ...         tracker.add_warning("Approaching limit")
        """
        tracker = CalculationTracker(
            calculation_type=calculation_type,
            formula_id=formula_id or self.CALCULATION_TYPES.get(calculation_type, "UNKNOWN"),
            algorithm=algorithm or calculation_type,
            parent_id=parent_id,
            random_seed=random_seed,
        )

        start_time = time.time()
        start_memory = self._get_memory_usage()

        try:
            yield tracker
            tracker.mark_success()
        except Exception as e:
            tracker.mark_failure(str(e))
            raise
        finally:
            # Calculate performance metrics
            end_time = time.time()
            calculation_time_ms = (end_time - start_time) * 1000

            end_memory = self._get_memory_usage()
            memory_usage = None
            if start_memory is not None and end_memory is not None:
                memory_usage = end_memory - start_memory

            # Generate calculation ID
            with self._lock:
                self._calculation_count += 1
                calc_id = f"CALC-{self._calculation_count:08d}-{uuid.uuid4().hex[:8]}"

            # Build calculation event
            event = self._build_event(
                tracker=tracker,
                calc_id=calc_id,
                calculation_time_ms=calculation_time_ms,
                memory_usage=memory_usage,
            )

            # Log to audit system
            self._log_calculation_event(event)

    def _build_event(
        self,
        tracker: "CalculationTracker",
        calc_id: str,
        calculation_time_ms: float,
        memory_usage: Optional[float],
    ) -> CalculationAuditEvent:
        """Build calculation audit event from tracker."""
        # Compute hashes
        input_hash = self._compute_hash(tracker.inputs)
        output_hash = self._compute_hash(tracker.outputs)

        # Compute provenance hash (chain of input hash + output hash)
        provenance_data = {
            "input_hash": input_hash,
            "output_hash": output_hash,
            "formula_id": tracker.formula_id,
            "formula_version": self.VERSION,
            "parent_id": tracker.parent_id,
        }
        provenance_hash = self._compute_hash(provenance_data)

        return CalculationAuditEvent(
            calculation_id=calc_id,
            calculation_type=tracker.calculation_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_summary=self._summarize_data(tracker.inputs),
            input_hash=input_hash,
            input_count=self._count_inputs(tracker.inputs),
            output_summary=self._summarize_data(tracker.outputs),
            output_hash=output_hash,
            formula_id=tracker.formula_id,
            formula_version=self.VERSION,
            algorithm=tracker.algorithm,
            is_deterministic=tracker.random_seed is None,
            random_seed=tracker.random_seed,
            solver_used=tracker.solver_used,
            iterations=tracker.iterations,
            calculation_time_ms=round(calculation_time_ms, 3),
            memory_usage_mb=round(memory_usage, 2) if memory_usage else None,
            success=tracker.success,
            error_message=tracker.error_message,
            warnings=tracker.warnings,
            parent_calculation_id=tracker.parent_id,
            provenance_hash=provenance_hash,
        )

    def _summarize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of data for audit (redact large arrays)."""
        if not data:
            return {}

        summary = {}
        for key, value in data.items():
            if isinstance(value, (list, tuple)):
                summary[key] = f"<{len(value)} items>"
            elif isinstance(value, dict):
                summary[key] = f"<dict with {len(value)} keys>"
            elif isinstance(value, (int, float, str, bool)):
                summary[key] = value
            else:
                summary[key] = f"<{type(value).__name__}>"
        return summary

    def _count_inputs(self, data: Dict[str, Any]) -> int:
        """Count total input elements."""
        if not data:
            return 0

        count = 0
        for value in data.values():
            if isinstance(value, (list, tuple)):
                count += len(value)
            elif isinstance(value, dict):
                count += len(value)
            else:
                count += 1
        return count

    def _log_calculation_event(self, event: CalculationAuditEvent) -> None:
        """Log calculation event to audit system."""
        # Create audit event
        severity = AuditSeverity.INFO if event.success else AuditSeverity.ERROR
        outcome = AuditOutcome.SUCCESS if event.success else AuditOutcome.FAILURE

        audit_event = AuditEvent(
            event_type=AuditEventType.OPTIMIZATION_RUN,
            action=f"calculate_{event.calculation_type}",
            severity=severity,
            outcome=outcome,
            message=(
                f"Calculation {event.calculation_id} ({event.calculation_type}): "
                f"{'completed' if event.success else 'failed'} in {event.calculation_time_ms:.1f}ms"
            ),
            details=event.to_dict(),
            metadata={
                "formula_id": event.formula_id,
                "input_hash": event.input_hash,
                "output_hash": event.output_hash,
                "provenance_hash": event.provenance_hash,
            },
        )

        self.audit_logger.log(audit_event)

    def log_pinch_analysis(
        self,
        input_streams: int,
        delta_t_min: float,
        pinch_temperature: float,
        min_hot_utility: float,
        min_cold_utility: float,
        max_heat_recovery: float,
        calculation_time_ms: float,
    ) -> AuditRecord:
        """
        Log pinch analysis calculation.

        Convenience method for pinch analysis results.
        """
        return self.audit_logger.log(AuditEvent(
            event_type=AuditEventType.OPTIMIZATION_RUN,
            action="calculate_pinch_analysis",
            severity=AuditSeverity.INFO,
            outcome=AuditOutcome.SUCCESS,
            message=f"Pinch analysis: T_pinch={pinch_temperature}C, Qh_min={min_hot_utility}kW",
            details={
                "calculation_type": "pinch_analysis",
                "formula_id": "PINCH_v1.0",
                "inputs": {
                    "stream_count": input_streams,
                    "delta_t_min_C": delta_t_min,
                },
                "outputs": {
                    "pinch_temperature_C": pinch_temperature,
                    "min_hot_utility_kW": min_hot_utility,
                    "min_cold_utility_kW": min_cold_utility,
                    "max_heat_recovery_kW": max_heat_recovery,
                },
                "calculation_time_ms": calculation_time_ms,
                "deterministic": True,
            },
        ))

    def log_lmtd_calculation(
        self,
        exchanger_id: str,
        duty_kw: float,
        lmtd: float,
        ua_value: float,
        area_m2: float,
        calculation_time_ms: float,
    ) -> AuditRecord:
        """Log LMTD calculation."""
        return self.audit_logger.log(AuditEvent(
            event_type=AuditEventType.OPTIMIZATION_RUN,
            action="calculate_lmtd",
            severity=AuditSeverity.INFO,
            outcome=AuditOutcome.SUCCESS,
            message=f"LMTD calculation for {exchanger_id}: LMTD={lmtd:.2f}C, A={area_m2:.2f}m2",
            details={
                "calculation_type": "lmtd_calculation",
                "formula_id": "LMTD_v1.0",
                "exchanger_id": exchanger_id,
                "outputs": {
                    "duty_kW": duty_kw,
                    "LMTD_C": lmtd,
                    "UA_kW_K": ua_value,
                    "area_m2": area_m2,
                },
                "calculation_time_ms": calculation_time_ms,
                "deterministic": True,
            },
        ))

    def log_exergy_analysis(
        self,
        design_id: str,
        total_exergy_input: float,
        total_exergy_destruction: float,
        exergy_efficiency: float,
        calculation_time_ms: float,
    ) -> AuditRecord:
        """Log exergy analysis calculation."""
        return self.audit_logger.log(AuditEvent(
            event_type=AuditEventType.OPTIMIZATION_RUN,
            action="calculate_exergy_analysis",
            severity=AuditSeverity.INFO,
            outcome=AuditOutcome.SUCCESS,
            message=f"Exergy analysis for {design_id}: efficiency={exergy_efficiency:.1%}",
            details={
                "calculation_type": "exergy_analysis",
                "formula_id": "EXERGY_v1.0",
                "design_id": design_id,
                "outputs": {
                    "total_exergy_input_kW": total_exergy_input,
                    "total_exergy_destruction_kW": total_exergy_destruction,
                    "exergy_efficiency": exergy_efficiency,
                },
                "calculation_time_ms": calculation_time_ms,
                "deterministic": True,
            },
        ))

    def log_economic_analysis(
        self,
        design_id: str,
        total_capital_cost: float,
        annual_operating_cost: float,
        annual_savings: float,
        payback_period: float,
        npv: float,
        calculation_time_ms: float,
    ) -> AuditRecord:
        """Log economic analysis calculation."""
        return self.audit_logger.log(AuditEvent(
            event_type=AuditEventType.OPTIMIZATION_RUN,
            action="calculate_economic_analysis",
            severity=AuditSeverity.INFO,
            outcome=AuditOutcome.SUCCESS,
            message=f"Economic analysis for {design_id}: payback={payback_period:.1f}yr, NPV=${npv:,.0f}",
            details={
                "calculation_type": "economic_analysis",
                "formula_id": "ECON_v1.0",
                "design_id": design_id,
                "outputs": {
                    "total_capital_cost_usd": total_capital_cost,
                    "annual_operating_cost_usd": annual_operating_cost,
                    "annual_savings_usd": annual_savings,
                    "payback_period_years": payback_period,
                    "npv_usd": npv,
                },
                "calculation_time_ms": calculation_time_ms,
                "deterministic": True,
            },
        ))

    def log_milp_optimization(
        self,
        optimization_id: str,
        solver: str,
        objective_value: float,
        iterations: int,
        solve_time_seconds: float,
        gap_percent: float,
        status: str,
    ) -> AuditRecord:
        """Log MILP optimization run."""
        return self.audit_logger.log(AuditEvent(
            event_type=AuditEventType.OPTIMIZATION_RUN,
            action="run_milp_optimization",
            severity=AuditSeverity.INFO if status == "optimal" else AuditSeverity.WARNING,
            outcome=AuditOutcome.SUCCESS if status in ["optimal", "feasible"] else AuditOutcome.PARTIAL,
            message=f"MILP optimization {optimization_id}: {status}, obj={objective_value:.2f}",
            details={
                "calculation_type": "milp_optimization",
                "formula_id": "MILP_HEN_v1.0",
                "optimization_id": optimization_id,
                "solver": solver,
                "outputs": {
                    "objective_value": objective_value,
                    "iterations": iterations,
                    "solve_time_seconds": solve_time_seconds,
                    "gap_percent": gap_percent,
                    "status": status,
                },
                "deterministic": True,
            },
        ))

    def log_hen_synthesis(
        self,
        design_id: str,
        exchanger_count: int,
        total_area_m2: float,
        heat_recovered_kw: float,
        hot_utility_kw: float,
        cold_utility_kw: float,
        calculation_time_ms: float,
    ) -> AuditRecord:
        """Log HEN synthesis result."""
        return self.audit_logger.log(AuditEvent(
            event_type=AuditEventType.OPTIMIZATION_RUN,
            action="synthesize_hen",
            severity=AuditSeverity.INFO,
            outcome=AuditOutcome.SUCCESS,
            message=f"HEN synthesis {design_id}: {exchanger_count} exchangers, {heat_recovered_kw:.0f}kW recovered",
            details={
                "calculation_type": "hen_synthesis",
                "formula_id": "HEN_SYNTH_v1.0",
                "design_id": design_id,
                "outputs": {
                    "exchanger_count": exchanger_count,
                    "total_area_m2": total_area_m2,
                    "heat_recovered_kW": heat_recovered_kw,
                    "hot_utility_required_kW": hot_utility_kw,
                    "cold_utility_required_kW": cold_utility_kw,
                },
                "calculation_time_ms": calculation_time_ms,
                "deterministic": True,
            },
        ))


class CalculationTracker:
    """
    Tracker for individual calculation events.

    Used with CalculationEventLogger context manager.
    """

    def __init__(
        self,
        calculation_type: str,
        formula_id: str,
        algorithm: str,
        parent_id: Optional[str] = None,
        random_seed: Optional[int] = None,
    ):
        self.calculation_type = calculation_type
        self.formula_id = formula_id
        self.algorithm = algorithm
        self.parent_id = parent_id
        self.random_seed = random_seed

        self.inputs: Dict[str, Any] = {}
        self.outputs: Dict[str, Any] = {}
        self.warnings: List[str] = []
        self.success: bool = False
        self.error_message: Optional[str] = None
        self.solver_used: Optional[str] = None
        self.iterations: Optional[int] = None

    def set_inputs(self, inputs: Dict[str, Any]) -> None:
        """Set calculation inputs."""
        self.inputs = inputs

    def set_outputs(self, outputs: Dict[str, Any]) -> None:
        """Set calculation outputs."""
        self.outputs = outputs

    def set_solver(self, solver: str, iterations: Optional[int] = None) -> None:
        """Set solver information."""
        self.solver_used = solver
        self.iterations = iterations

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)

    def mark_success(self) -> None:
        """Mark calculation as successful."""
        self.success = True
        self.error_message = None

    def mark_failure(self, error_message: str) -> None:
        """Mark calculation as failed."""
        self.success = False
        self.error_message = error_message


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_audit_logger() -> AuditLogger:
    """Get the singleton audit logger instance."""
    return AuditLogger()


def get_calculation_logger() -> CalculationEventLogger:
    """Get a calculation event logger."""
    return CalculationEventLogger(get_audit_logger())
