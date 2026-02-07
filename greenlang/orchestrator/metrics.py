# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-FOUND-001: GreenLang DAG Orchestrator

12 Prometheus metrics for DAG execution monitoring with graceful
fallback when prometheus_client is not installed.

Metrics:
    1.  gl_orchestrator_dag_executions_total (Counter)
    2.  gl_orchestrator_dag_execution_duration_seconds (Histogram)
    3.  gl_orchestrator_node_executions_total (Counter)
    4.  gl_orchestrator_node_execution_duration_seconds (Histogram)
    5.  gl_orchestrator_node_retries_total (Counter)
    6.  gl_orchestrator_node_timeouts_total (Counter)
    7.  gl_orchestrator_active_executions (Gauge)
    8.  gl_orchestrator_checkpoint_operations_total (Counter)
    9.  gl_orchestrator_checkpoint_size_bytes (Histogram)
    10. gl_orchestrator_provenance_chain_length (Histogram)
    11. gl_orchestrator_parallel_nodes_per_level (Histogram)
    12. gl_orchestrator_dag_validation_errors_total (Counter)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-001 GreenLang Orchestrator
Status: Production Ready
"""

from __future__ import annotations

import logging
from typing import Optional

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
        "prometheus_client not installed; orchestrator metrics disabled"
    )


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # 1. DAG execution count
    dag_executions_total = Counter(
        "gl_orchestrator_dag_executions_total",
        "Total DAG executions",
        labelnames=["dag_id", "status"],
    )

    # 2. DAG execution duration
    dag_execution_duration_seconds = Histogram(
        "gl_orchestrator_dag_execution_duration_seconds",
        "DAG execution duration in seconds",
        labelnames=["dag_id"],
        buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
    )

    # 3. Node execution count
    node_executions_total = Counter(
        "gl_orchestrator_node_executions_total",
        "Total node executions",
        labelnames=["dag_id", "node_id", "status"],
    )

    # 4. Node execution duration
    node_execution_duration_seconds = Histogram(
        "gl_orchestrator_node_execution_duration_seconds",
        "Node execution duration in seconds",
        labelnames=["dag_id", "node_id"],
        buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
    )

    # 5. Node retries
    node_retries_total = Counter(
        "gl_orchestrator_node_retries_total",
        "Total node retry attempts",
        labelnames=["dag_id", "node_id"],
    )

    # 6. Node timeouts
    node_timeouts_total = Counter(
        "gl_orchestrator_node_timeouts_total",
        "Total node timeout occurrences",
        labelnames=["dag_id", "node_id"],
    )

    # 7. Active executions
    active_executions = Gauge(
        "gl_orchestrator_active_executions",
        "Number of currently active DAG executions",
    )

    # 8. Checkpoint operations
    checkpoint_operations_total = Counter(
        "gl_orchestrator_checkpoint_operations_total",
        "Total checkpoint operations",
        labelnames=["operation"],
    )

    # 9. Checkpoint size
    checkpoint_size_bytes = Histogram(
        "gl_orchestrator_checkpoint_size_bytes",
        "Checkpoint data size in bytes",
        buckets=(
            100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000,
        ),
    )

    # 10. Provenance chain length
    provenance_chain_length = Histogram(
        "gl_orchestrator_provenance_chain_length",
        "Number of nodes in provenance chain",
        labelnames=["dag_id"],
        buckets=(1, 5, 10, 20, 50, 100, 200, 500),
    )

    # 11. Parallel nodes per level
    parallel_nodes_per_level = Histogram(
        "gl_orchestrator_parallel_nodes_per_level",
        "Number of parallel nodes per execution level",
        labelnames=["dag_id"],
        buckets=(1, 2, 3, 5, 10, 20, 50, 100),
    )

    # 12. DAG validation errors
    dag_validation_errors_total = Counter(
        "gl_orchestrator_dag_validation_errors_total",
        "Total DAG validation errors",
        labelnames=["error_type"],
    )

else:
    # No-op placeholders
    dag_executions_total = None  # type: ignore[assignment]
    dag_execution_duration_seconds = None  # type: ignore[assignment]
    node_executions_total = None  # type: ignore[assignment]
    node_execution_duration_seconds = None  # type: ignore[assignment]
    node_retries_total = None  # type: ignore[assignment]
    node_timeouts_total = None  # type: ignore[assignment]
    active_executions = None  # type: ignore[assignment]
    checkpoint_operations_total = None  # type: ignore[assignment]
    checkpoint_size_bytes = None  # type: ignore[assignment]
    provenance_chain_length = None  # type: ignore[assignment]
    parallel_nodes_per_level = None  # type: ignore[assignment]
    dag_validation_errors_total = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_dag_execution(
    dag_id: str, status: str, duration_seconds: float,
) -> None:
    """Record a DAG execution completion.

    Args:
        dag_id: DAG identifier.
        status: Final execution status.
        duration_seconds: Total duration in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    dag_executions_total.labels(dag_id=dag_id, status=status).inc()
    dag_execution_duration_seconds.labels(dag_id=dag_id).observe(
        duration_seconds,
    )


def record_node_execution(
    dag_id: str, node_id: str, status: str, duration_seconds: float,
) -> None:
    """Record a node execution completion.

    Args:
        dag_id: DAG identifier.
        node_id: Node identifier.
        status: Final node status.
        duration_seconds: Duration in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    node_executions_total.labels(
        dag_id=dag_id, node_id=node_id, status=status,
    ).inc()
    node_execution_duration_seconds.labels(
        dag_id=dag_id, node_id=node_id,
    ).observe(duration_seconds)


def record_node_retry(dag_id: str, node_id: str) -> None:
    """Record a node retry attempt.

    Args:
        dag_id: DAG identifier.
        node_id: Node identifier.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    node_retries_total.labels(dag_id=dag_id, node_id=node_id).inc()


def record_node_timeout(dag_id: str, node_id: str) -> None:
    """Record a node timeout.

    Args:
        dag_id: DAG identifier.
        node_id: Node identifier.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    node_timeouts_total.labels(dag_id=dag_id, node_id=node_id).inc()


def increment_active_executions() -> None:
    """Increment the active executions gauge."""
    if not PROMETHEUS_AVAILABLE:
        return
    active_executions.inc()


def decrement_active_executions() -> None:
    """Decrement the active executions gauge."""
    if not PROMETHEUS_AVAILABLE:
        return
    active_executions.dec()


def record_checkpoint_operation(
    operation: str, size_bytes: Optional[int] = None,
) -> None:
    """Record a checkpoint operation.

    Args:
        operation: Operation type (save, load, delete).
        size_bytes: Optional data size in bytes.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    checkpoint_operations_total.labels(operation=operation).inc()
    if size_bytes is not None:
        checkpoint_size_bytes.observe(size_bytes)


def record_provenance_chain(dag_id: str, length: int) -> None:
    """Record provenance chain length.

    Args:
        dag_id: DAG identifier.
        length: Number of nodes in provenance chain.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    provenance_chain_length.labels(dag_id=dag_id).observe(length)


def record_parallel_nodes(dag_id: str, count: int) -> None:
    """Record parallel nodes per level.

    Args:
        dag_id: DAG identifier.
        count: Number of parallel nodes in this level.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    parallel_nodes_per_level.labels(dag_id=dag_id).observe(count)


def record_validation_error(error_type: str) -> None:
    """Record a DAG validation error.

    Args:
        error_type: Type of validation error.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    dag_validation_errors_total.labels(error_type=error_type).inc()


__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "dag_executions_total",
    "dag_execution_duration_seconds",
    "node_executions_total",
    "node_execution_duration_seconds",
    "node_retries_total",
    "node_timeouts_total",
    "active_executions",
    "checkpoint_operations_total",
    "checkpoint_size_bytes",
    "provenance_chain_length",
    "parallel_nodes_per_level",
    "dag_validation_errors_total",
    # Helper functions
    "record_dag_execution",
    "record_node_execution",
    "record_node_retry",
    "record_node_timeout",
    "increment_active_executions",
    "decrement_active_executions",
    "record_checkpoint_operation",
    "record_provenance_chain",
    "record_parallel_nodes",
    "record_validation_error",
]
