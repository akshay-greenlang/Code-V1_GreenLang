# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-EUDR-026: Due Diligence Orchestrator

20 Prometheus metrics for due diligence orchestrator agent service monitoring
with graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_eudr_ddo_`` prefix (GreenLang EUDR Due Diligence
Orchestrator) for consistent identification in Prometheus queries, Grafana
dashboards, and alerting rules.

Metrics (20 per PRD):
    Counters (10):
        1.  gl_eudr_ddo_workflows_created_total          - Workflows created
        2.  gl_eudr_ddo_workflows_completed_total         - Workflows completed
        3.  gl_eudr_ddo_workflows_failed_total            - Workflows failed/terminated
        4.  gl_eudr_ddo_agent_executions_total            - Agent executions started
        5.  gl_eudr_ddo_agent_completions_total           - Agent executions completed
        6.  gl_eudr_ddo_quality_gate_evaluations_total    - Quality gate evaluations
        7.  gl_eudr_ddo_retry_attempts_total              - Retry attempts
        8.  gl_eudr_ddo_circuit_breaker_transitions_total - Circuit breaker transitions
        9.  gl_eudr_ddo_packages_generated_total          - DD packages generated
        10. gl_eudr_ddo_api_errors_total                  - API errors by operation

    Histograms (4):
        11. gl_eudr_ddo_workflow_duration_seconds          - Workflow execution duration
        12. gl_eudr_ddo_agent_execution_seconds            - Individual agent execution duration
        13. gl_eudr_ddo_quality_gate_seconds               - Quality gate evaluation duration
        14. gl_eudr_ddo_package_generation_seconds          - Package generation duration

    Gauges (6):
        15. gl_eudr_ddo_active_workflows                   - Currently active workflows
        16. gl_eudr_ddo_running_agents                     - Currently running agents
        17. gl_eudr_ddo_queued_agents                      - Agents waiting in queue
        18. gl_eudr_ddo_circuit_breakers_open               - Open circuit breakers
        19. gl_eudr_ddo_dead_letter_entries                 - Dead letter queue size
        20. gl_eudr_ddo_checkpoint_backlog                  - Checkpoint write backlog

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-026 Due Diligence Orchestrator (GL-EUDR-DDO-026)
Status: Production Ready
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful prometheus_client import
# ---------------------------------------------------------------------------

try:
    from prometheus_client import Counter, Gauge, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.info(
        "prometheus_client not installed; "
        "due diligence orchestrator metrics disabled"
    )

# ---------------------------------------------------------------------------
# Safe metric registration helpers
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    from prometheus_client import REGISTRY as _REGISTRY

    def _safe_counter(name: str, doc: str, labels: list) -> Counter:
        """Safely create or retrieve a Counter metric."""
        try:
            return Counter(name, doc, labels, registry=_REGISTRY)
        except ValueError:
            return _REGISTRY._names_to_collectors[name]

    def _safe_histogram(name: str, doc: str, labels: list, buckets: tuple) -> Histogram:
        """Safely create or retrieve a Histogram metric."""
        try:
            return Histogram(name, doc, labels, buckets=buckets, registry=_REGISTRY)
        except ValueError:
            return _REGISTRY._names_to_collectors[name]

    def _safe_gauge(name: str, doc: str, labels: list) -> Gauge:
        """Safely create or retrieve a Gauge metric."""
        try:
            return Gauge(name, doc, labels, registry=_REGISTRY)
        except ValueError:
            return _REGISTRY._names_to_collectors[name]

# ---------------------------------------------------------------------------
# Metric definitions (20 metrics)
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # Counters (10)
    _workflows_created_total = _safe_counter(
        "gl_eudr_ddo_workflows_created_total",
        "Total number of due diligence workflows created",
        ["workflow_type", "commodity"],
    )
    _workflows_completed_total = _safe_counter(
        "gl_eudr_ddo_workflows_completed_total",
        "Total number of workflows completed successfully",
        ["workflow_type", "commodity"],
    )
    _workflows_failed_total = _safe_counter(
        "gl_eudr_ddo_workflows_failed_total",
        "Total number of workflows failed or terminated",
        ["workflow_type", "reason"],
    )
    _agent_executions_total = _safe_counter(
        "gl_eudr_ddo_agent_executions_total",
        "Total number of agent executions started",
        ["agent_id", "phase"],
    )
    _agent_completions_total = _safe_counter(
        "gl_eudr_ddo_agent_completions_total",
        "Total number of agent executions completed",
        ["agent_id", "status"],
    )
    _quality_gate_evaluations_total = _safe_counter(
        "gl_eudr_ddo_quality_gate_evaluations_total",
        "Total number of quality gate evaluations",
        ["gate_id", "result"],
    )
    _retry_attempts_total = _safe_counter(
        "gl_eudr_ddo_retry_attempts_total",
        "Total number of agent retry attempts",
        ["agent_id", "error_type"],
    )
    _circuit_breaker_transitions_total = _safe_counter(
        "gl_eudr_ddo_circuit_breaker_transitions_total",
        "Total circuit breaker state transitions",
        ["agent_id", "from_state", "to_state"],
    )
    _packages_generated_total = _safe_counter(
        "gl_eudr_ddo_packages_generated_total",
        "Total due diligence packages generated",
        ["workflow_type", "language"],
    )
    _api_errors_total = _safe_counter(
        "gl_eudr_ddo_api_errors_total",
        "Total API errors by operation",
        ["operation"],
    )

    # Histograms (4)
    _workflow_duration_seconds = _safe_histogram(
        "gl_eudr_ddo_workflow_duration_seconds",
        "Workflow total execution duration in seconds",
        ["workflow_type"],
        buckets=(10, 30, 60, 120, 180, 300, 600, 900, 1800, 3600),
    )
    _agent_execution_seconds = _safe_histogram(
        "gl_eudr_ddo_agent_execution_seconds",
        "Individual agent execution duration in seconds",
        ["agent_id"],
        buckets=(0.5, 1, 2.5, 5, 10, 15, 30, 60, 120, 300),
    )
    _quality_gate_seconds = _safe_histogram(
        "gl_eudr_ddo_quality_gate_seconds",
        "Quality gate evaluation duration in seconds",
        ["gate_id"],
        buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 15, 30, 60),
    )
    _package_generation_seconds = _safe_histogram(
        "gl_eudr_ddo_package_generation_seconds",
        "Due diligence package generation duration in seconds",
        ["workflow_type"],
        buckets=(1, 2.5, 5, 10, 15, 30, 60, 120, 300, 600),
    )

    # Gauges (6)
    _active_workflows = _safe_gauge(
        "gl_eudr_ddo_active_workflows",
        "Number of currently active workflows",
        ["workflow_type"],
    )
    _running_agents = _safe_gauge(
        "gl_eudr_ddo_running_agents",
        "Number of agents currently executing",
        ["phase"],
    )
    _queued_agents = _safe_gauge(
        "gl_eudr_ddo_queued_agents",
        "Number of agents waiting in execution queue",
        ["phase"],
    )
    _circuit_breakers_open = _safe_gauge(
        "gl_eudr_ddo_circuit_breakers_open",
        "Number of open circuit breakers",
        [],
    )
    _dead_letter_entries = _safe_gauge(
        "gl_eudr_ddo_dead_letter_entries",
        "Number of entries in dead letter queue",
        [],
    )
    _checkpoint_backlog = _safe_gauge(
        "gl_eudr_ddo_checkpoint_backlog",
        "Number of checkpoints pending write",
        [],
    )


# ---------------------------------------------------------------------------
# Helper functions (20 functions matching 20 metrics)
# ---------------------------------------------------------------------------


def record_workflow_created(
    workflow_type: str = "standard",
    commodity: str = "unknown",
) -> None:
    """Record a workflow creation event.

    Args:
        workflow_type: Type of workflow (standard/simplified/custom).
        commodity: EUDR commodity being assessed.
    """
    if PROMETHEUS_AVAILABLE:
        _workflows_created_total.labels(
            workflow_type=workflow_type,
            commodity=commodity,
        ).inc()


def record_workflow_completed(
    workflow_type: str = "standard",
    commodity: str = "unknown",
) -> None:
    """Record a workflow completion event.

    Args:
        workflow_type: Type of workflow.
        commodity: EUDR commodity assessed.
    """
    if PROMETHEUS_AVAILABLE:
        _workflows_completed_total.labels(
            workflow_type=workflow_type,
            commodity=commodity,
        ).inc()


def record_workflow_failed(
    workflow_type: str = "standard",
    reason: str = "unknown",
) -> None:
    """Record a workflow failure event.

    Args:
        workflow_type: Type of workflow.
        reason: Failure reason category.
    """
    if PROMETHEUS_AVAILABLE:
        _workflows_failed_total.labels(
            workflow_type=workflow_type,
            reason=reason,
        ).inc()


def record_agent_execution(
    agent_id: str = "unknown",
    phase: str = "unknown",
) -> None:
    """Record an agent execution start.

    Args:
        agent_id: EUDR agent identifier.
        phase: Due diligence phase.
    """
    if PROMETHEUS_AVAILABLE:
        _agent_executions_total.labels(
            agent_id=agent_id,
            phase=phase,
        ).inc()


def record_agent_completion(
    agent_id: str = "unknown",
    status: str = "completed",
) -> None:
    """Record an agent execution completion.

    Args:
        agent_id: EUDR agent identifier.
        status: Completion status (completed/failed/skipped).
    """
    if PROMETHEUS_AVAILABLE:
        _agent_completions_total.labels(
            agent_id=agent_id,
            status=status,
        ).inc()


def record_quality_gate_evaluation(
    gate_id: str = "QG-1",
    result: str = "pending",
) -> None:
    """Record a quality gate evaluation.

    Args:
        gate_id: Quality gate identifier.
        result: Evaluation result (passed/failed/overridden).
    """
    if PROMETHEUS_AVAILABLE:
        _quality_gate_evaluations_total.labels(
            gate_id=gate_id,
            result=result,
        ).inc()


def record_retry_attempt(
    agent_id: str = "unknown",
    error_type: str = "transient",
) -> None:
    """Record an agent retry attempt.

    Args:
        agent_id: Agent being retried.
        error_type: Type of error triggering retry.
    """
    if PROMETHEUS_AVAILABLE:
        _retry_attempts_total.labels(
            agent_id=agent_id,
            error_type=error_type,
        ).inc()


def record_circuit_breaker_transition(
    agent_id: str = "unknown",
    from_state: str = "closed",
    to_state: str = "open",
) -> None:
    """Record a circuit breaker state transition.

    Args:
        agent_id: Agent with circuit breaker.
        from_state: Previous state.
        to_state: New state.
    """
    if PROMETHEUS_AVAILABLE:
        _circuit_breaker_transitions_total.labels(
            agent_id=agent_id,
            from_state=from_state,
            to_state=to_state,
        ).inc()


def record_package_generated(
    workflow_type: str = "standard",
    language: str = "en",
) -> None:
    """Record a due diligence package generation.

    Args:
        workflow_type: Type of workflow.
        language: Package language.
    """
    if PROMETHEUS_AVAILABLE:
        _packages_generated_total.labels(
            workflow_type=workflow_type,
            language=language,
        ).inc()


def record_api_error(operation: str) -> None:
    """Record an API error.

    Args:
        operation: Operation name.
    """
    if PROMETHEUS_AVAILABLE:
        _api_errors_total.labels(operation=operation).inc()


def observe_workflow_duration(
    duration_seconds: float,
    workflow_type: str = "standard",
) -> None:
    """Observe workflow execution duration.

    Args:
        duration_seconds: Duration in seconds.
        workflow_type: Type of workflow.
    """
    if PROMETHEUS_AVAILABLE:
        _workflow_duration_seconds.labels(
            workflow_type=workflow_type,
        ).observe(duration_seconds)


def observe_agent_execution_duration(
    duration_seconds: float,
    agent_id: str = "unknown",
) -> None:
    """Observe individual agent execution duration.

    Args:
        duration_seconds: Duration in seconds.
        agent_id: EUDR agent identifier.
    """
    if PROMETHEUS_AVAILABLE:
        _agent_execution_seconds.labels(
            agent_id=agent_id,
        ).observe(duration_seconds)


def observe_quality_gate_duration(
    duration_seconds: float,
    gate_id: str = "QG-1",
) -> None:
    """Observe quality gate evaluation duration.

    Args:
        duration_seconds: Duration in seconds.
        gate_id: Quality gate identifier.
    """
    if PROMETHEUS_AVAILABLE:
        _quality_gate_seconds.labels(
            gate_id=gate_id,
        ).observe(duration_seconds)


def observe_package_generation_duration(
    duration_seconds: float,
    workflow_type: str = "standard",
) -> None:
    """Observe package generation duration.

    Args:
        duration_seconds: Duration in seconds.
        workflow_type: Type of workflow.
    """
    if PROMETHEUS_AVAILABLE:
        _package_generation_seconds.labels(
            workflow_type=workflow_type,
        ).observe(duration_seconds)


def set_active_workflows(
    count: int,
    workflow_type: str = "all",
) -> None:
    """Set the number of active workflows.

    Args:
        count: Number of active workflows.
        workflow_type: Workflow type filter.
    """
    if PROMETHEUS_AVAILABLE:
        _active_workflows.labels(workflow_type=workflow_type).set(count)


def set_running_agents(
    count: int,
    phase: str = "all",
) -> None:
    """Set the number of currently running agents.

    Args:
        count: Number of running agents.
        phase: Phase filter.
    """
    if PROMETHEUS_AVAILABLE:
        _running_agents.labels(phase=phase).set(count)


def set_queued_agents(
    count: int,
    phase: str = "all",
) -> None:
    """Set the number of queued agents.

    Args:
        count: Number of queued agents.
        phase: Phase filter.
    """
    if PROMETHEUS_AVAILABLE:
        _queued_agents.labels(phase=phase).set(count)


def set_circuit_breakers_open(count: int) -> None:
    """Set the number of open circuit breakers.

    Args:
        count: Number of open circuit breakers.
    """
    if PROMETHEUS_AVAILABLE:
        _circuit_breakers_open.set(count)


def set_dead_letter_entries(count: int) -> None:
    """Set the number of dead letter queue entries.

    Args:
        count: Number of dead letter entries.
    """
    if PROMETHEUS_AVAILABLE:
        _dead_letter_entries.set(count)


def set_checkpoint_backlog(count: int) -> None:
    """Set the checkpoint write backlog size.

    Args:
        count: Number of pending checkpoint writes.
    """
    if PROMETHEUS_AVAILABLE:
        _checkpoint_backlog.set(count)
