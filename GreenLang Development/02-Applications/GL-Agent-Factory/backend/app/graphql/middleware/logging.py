"""
GraphQL Logging Middleware

Implements comprehensive logging for the GreenLang GraphQL API.
Provides request logging, audit trails, and performance metrics.

Features:
- Request/response logging
- Operation timing
- Error tracking
- Audit trail for compliance
- Prometheus metrics

Example:
    schema = Schema(
        query=Query,
        mutation=Mutation,
        extensions=[LoggingMiddleware],
    )
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from strawberry.extensions import Extension

logger = logging.getLogger(__name__)

# Separate logger for audit events
audit_logger = logging.getLogger("greenlang.audit")


# =============================================================================
# Metrics Collection
# =============================================================================


@dataclass
class GraphQLMetrics:
    """
    Metrics collector for GraphQL operations.

    Tracks request counts, latencies, and error rates.
    """

    # Counters
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    # Latency tracking (in seconds)
    total_latency: float = 0.0
    min_latency: float = float("inf")
    max_latency: float = 0.0

    # Operation counts
    queries: int = 0
    mutations: int = 0
    subscriptions: int = 0

    # Error tracking
    errors_by_type: Dict[str, int] = field(default_factory=dict)

    # Per-operation metrics
    operation_counts: Dict[str, int] = field(default_factory=dict)
    operation_latencies: Dict[str, List[float]] = field(default_factory=dict)

    def record_request(
        self,
        operation_name: Optional[str],
        operation_type: str,
        latency: float,
        success: bool,
        error_type: Optional[str] = None,
    ) -> None:
        """Record metrics for a completed request."""
        self.total_requests += 1
        self.total_latency += latency
        self.min_latency = min(self.min_latency, latency)
        self.max_latency = max(self.max_latency, latency)

        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            if error_type:
                self.errors_by_type[error_type] = self.errors_by_type.get(error_type, 0) + 1

        # Track by operation type
        if operation_type == "query":
            self.queries += 1
        elif operation_type == "mutation":
            self.mutations += 1
        elif operation_type == "subscription":
            self.subscriptions += 1

        # Track by operation name
        if operation_name:
            self.operation_counts[operation_name] = (
                self.operation_counts.get(operation_name, 0) + 1
            )
            if operation_name not in self.operation_latencies:
                self.operation_latencies[operation_name] = []
            self.operation_latencies[operation_name].append(latency)

    @property
    def average_latency(self) -> float:
        """Calculate average latency."""
        if self.total_requests == 0:
            return 0.0
        return self.total_latency / self.total_requests

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests

    def get_operation_p95_latency(self, operation_name: str) -> float:
        """Get p95 latency for a specific operation."""
        latencies = self.operation_latencies.get(operation_name, [])
        if not latencies:
            return 0.0
        sorted_latencies = sorted(latencies)
        p95_index = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[p95_index] if p95_index < len(sorted_latencies) else sorted_latencies[-1]

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.success_rate,
            "average_latency_ms": self.average_latency * 1000,
            "min_latency_ms": self.min_latency * 1000 if self.min_latency != float("inf") else 0,
            "max_latency_ms": self.max_latency * 1000,
            "queries": self.queries,
            "mutations": self.mutations,
            "subscriptions": self.subscriptions,
            "errors_by_type": self.errors_by_type,
            "operation_counts": self.operation_counts,
        }


# Global metrics instance
_metrics = GraphQLMetrics()


def get_metrics() -> GraphQLMetrics:
    """Get global metrics instance."""
    return _metrics


# =============================================================================
# Request Logger
# =============================================================================


class RequestLogger:
    """
    Logger for GraphQL requests.

    Provides structured logging with context information.
    """

    def __init__(self, request_id: str):
        self.request_id = request_id
        self.start_time = time.time()
        self.context: Dict[str, Any] = {}

    def set_context(self, **kwargs) -> None:
        """Set additional context for logging."""
        self.context.update(kwargs)

    def log_start(
        self,
        operation_name: Optional[str],
        operation_type: str,
        variables: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log request start."""
        logger.info(
            f"GraphQL request started",
            extra={
                "request_id": self.request_id,
                "operation_name": operation_name,
                "operation_type": operation_type,
                "has_variables": variables is not None,
                **self.context,
            },
        )

    def log_end(
        self,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        """Log request completion."""
        duration = time.time() - self.start_time

        log_data = {
            "request_id": self.request_id,
            "duration_ms": duration * 1000,
            "success": success,
            **self.context,
        }

        if error:
            log_data["error"] = error

        if success:
            logger.info(f"GraphQL request completed", extra=log_data)
        else:
            logger.error(f"GraphQL request failed", extra=log_data)

    def log_error(self, error: Exception) -> None:
        """Log an error during request processing."""
        logger.error(
            f"GraphQL error: {error}",
            extra={
                "request_id": self.request_id,
                "error_type": type(error).__name__,
                **self.context,
            },
            exc_info=True,
        )


# =============================================================================
# Audit Logger
# =============================================================================


class AuditLogger:
    """
    Audit logger for compliance requirements.

    Records all data-modifying operations with full context.
    """

    @staticmethod
    def log_mutation(
        request_id: str,
        operation_name: str,
        user_id: Optional[str],
        tenant_id: str,
        input_data: Dict[str, Any],
        output_data: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """
        Log a mutation operation for audit.

        Args:
            request_id: Request identifier
            operation_name: Name of the mutation
            user_id: User who performed the operation
            tenant_id: Tenant context
            input_data: Mutation input (sanitized)
            output_data: Mutation output (sanitized)
            success: Whether operation succeeded
            error: Error message if failed
        """
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "graphql_mutation",
            "request_id": request_id,
            "operation_name": operation_name,
            "user_id": user_id,
            "tenant_id": tenant_id,
            "input_summary": _sanitize_for_audit(input_data),
            "success": success,
        }

        if output_data:
            audit_entry["output_summary"] = _sanitize_for_audit(output_data)

        if error:
            audit_entry["error"] = error

        audit_logger.info(
            f"AUDIT: {operation_name}",
            extra=audit_entry,
        )

    @staticmethod
    def log_access(
        request_id: str,
        operation_name: str,
        user_id: Optional[str],
        tenant_id: str,
        resource_type: str,
        resource_id: Optional[str] = None,
    ) -> None:
        """
        Log a data access operation for audit.

        Args:
            request_id: Request identifier
            operation_name: Name of the query
            user_id: User who accessed the data
            tenant_id: Tenant context
            resource_type: Type of resource accessed
            resource_id: Specific resource ID if applicable
        """
        audit_logger.info(
            f"AUDIT ACCESS: {operation_name}",
            extra={
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": "graphql_access",
                "request_id": request_id,
                "operation_name": operation_name,
                "user_id": user_id,
                "tenant_id": tenant_id,
                "resource_type": resource_type,
                "resource_id": resource_id,
            },
        )


def _sanitize_for_audit(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize data for audit logging.

    Removes sensitive fields and truncates large values.
    """
    sensitive_fields = {
        "password", "secret", "token", "api_key", "apikey",
        "authorization", "auth", "credential", "private_key",
    }

    sanitized = {}
    for key, value in data.items():
        key_lower = key.lower()

        # Skip sensitive fields
        if any(s in key_lower for s in sensitive_fields):
            sanitized[key] = "[REDACTED]"
            continue

        # Truncate large strings
        if isinstance(value, str) and len(value) > 500:
            sanitized[key] = value[:500] + "...[TRUNCATED]"
        # Truncate large lists
        elif isinstance(value, list) and len(value) > 10:
            sanitized[key] = f"[LIST: {len(value)} items]"
        # Recurse into dicts
        elif isinstance(value, dict):
            sanitized[key] = _sanitize_for_audit(value)
        else:
            sanitized[key] = value

    return sanitized


# =============================================================================
# Strawberry Extension
# =============================================================================


class LoggingMiddleware(Extension):
    """
    Strawberry extension for comprehensive logging.

    Provides:
    - Request/response logging
    - Performance metrics
    - Audit trails for mutations
    - Error tracking
    """

    def __init__(self):
        self.request_logger: Optional[RequestLogger] = None
        self.start_time: float = 0
        self.operation_name: Optional[str] = None
        self.operation_type: str = "unknown"

    async def on_request_start(self) -> None:
        """Called at the start of each request."""
        self.start_time = time.time()

        # Generate request ID
        request_id = str(uuid.uuid4())

        # Get context info
        context = self.execution_context.context
        tenant_id = context.get("tenant_id", "unknown")
        user_id = context.get("user_id")

        # Create request logger
        self.request_logger = RequestLogger(request_id)
        self.request_logger.set_context(
            tenant_id=tenant_id,
            user_id=user_id,
        )

        # Store request_id in context for other components
        context["request_id"] = request_id

    async def on_request_end(self) -> None:
        """Called at the end of each request."""
        if not self.request_logger:
            return

        duration = time.time() - self.start_time
        result = self.execution_context.result

        # Determine success
        success = True
        error_type = None
        error_message = None

        if result and hasattr(result, "errors") and result.errors:
            success = False
            error_type = type(result.errors[0]).__name__
            error_message = str(result.errors[0])

        # Log completion
        self.request_logger.log_end(success, error_message)

        # Record metrics
        _metrics.record_request(
            operation_name=self.operation_name,
            operation_type=self.operation_type,
            latency=duration,
            success=success,
            error_type=error_type,
        )

    def on_operation(self) -> None:
        """Called before executing an operation."""
        # Extract operation info
        operation = self.execution_context.operation_name
        operation_type = self.execution_context.operation_type

        self.operation_name = operation
        self.operation_type = operation_type.value if hasattr(operation_type, "value") else str(operation_type)

        if self.request_logger:
            self.request_logger.set_context(
                operation_name=operation,
                operation_type=self.operation_type,
            )
            self.request_logger.log_start(
                operation_name=operation,
                operation_type=self.operation_type,
            )

    def on_validate(self) -> None:
        """Called during validation."""
        pass

    def on_parse(self) -> None:
        """Called during parsing."""
        pass

    def resolve(self, _next, root, info, *args, **kwargs):
        """Called for each field resolution."""
        return _next(root, info, *args, **kwargs)


# =============================================================================
# Prometheus Metrics Export
# =============================================================================


def get_prometheus_metrics() -> str:
    """
    Export metrics in Prometheus format.

    Returns:
        Prometheus-formatted metrics string
    """
    metrics = get_metrics()

    lines = [
        "# HELP graphql_requests_total Total number of GraphQL requests",
        "# TYPE graphql_requests_total counter",
        f"graphql_requests_total {metrics.total_requests}",
        "",
        "# HELP graphql_requests_success_total Successful GraphQL requests",
        "# TYPE graphql_requests_success_total counter",
        f"graphql_requests_success_total {metrics.successful_requests}",
        "",
        "# HELP graphql_requests_failed_total Failed GraphQL requests",
        "# TYPE graphql_requests_failed_total counter",
        f"graphql_requests_failed_total {metrics.failed_requests}",
        "",
        "# HELP graphql_request_duration_seconds Request duration in seconds",
        "# TYPE graphql_request_duration_seconds summary",
        f'graphql_request_duration_seconds{{quantile="0.5"}} {metrics.average_latency}',
        "",
        "# HELP graphql_queries_total Total number of queries",
        "# TYPE graphql_queries_total counter",
        f"graphql_queries_total {metrics.queries}",
        "",
        "# HELP graphql_mutations_total Total number of mutations",
        "# TYPE graphql_mutations_total counter",
        f"graphql_mutations_total {metrics.mutations}",
        "",
        "# HELP graphql_subscriptions_total Total number of subscriptions",
        "# TYPE graphql_subscriptions_total counter",
        f"graphql_subscriptions_total {metrics.subscriptions}",
    ]

    # Add per-operation metrics
    for op_name, count in metrics.operation_counts.items():
        lines.extend([
            "",
            f'graphql_operation_requests_total{{operation="{op_name}"}} {count}',
        ])

    return "\n".join(lines)
