# -*- coding: utf-8 -*-
"""
Agent Factory Telemetry - INFRA-010 Phase 4

OpenTelemetry-based distributed tracing, correlation propagation,
and Prometheus metrics collection for the GreenLang Agent Factory.

Public API:
    - AgentTracer: Singleton OTLP tracer wrapper for agent execution.
    - AgentSpan / SpanFactory: Typed span builders for agent operations.
    - CorrelationManager: Correlation-ID propagation across agents.
    - TelemetryExporter: Multi-backend span exporter with batch flush.
    - AgentMetricsCollector: Prometheus counters, histograms, and gauges.

Example:
    >>> from greenlang.infrastructure.agent_factory.telemetry import (
    ...     AgentTracer, SpanFactory, CorrelationManager,
    ...     TelemetryExporter, AgentMetricsCollector,
    ... )
    >>> tracer = AgentTracer.get_instance()
    >>> span = SpanFactory.execution_span("carbon-calc", "2.1.0")

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from greenlang.infrastructure.agent_factory.telemetry.tracer import (
    AgentTracer,
    TracerConfig,
)
from greenlang.infrastructure.agent_factory.telemetry.spans import (
    AgentExecutionSpan,
    LifecycleSpan,
    QueueSpan,
    MessageSpan,
    SpanFactory,
)
from greenlang.infrastructure.agent_factory.telemetry.correlation import (
    CorrelationContext,
    CorrelationManager,
)
from greenlang.infrastructure.agent_factory.telemetry.exporter import (
    ExporterConfig,
    TelemetryExporter,
)
from greenlang.infrastructure.agent_factory.telemetry.metrics_collector import (
    AgentMetricsCollector,
    MetricPoint,
)

__all__ = [
    "AgentExecutionSpan",
    "AgentMetricsCollector",
    "AgentTracer",
    "CorrelationContext",
    "CorrelationManager",
    "ExporterConfig",
    "LifecycleSpan",
    "MessageSpan",
    "MetricPoint",
    "QueueSpan",
    "SpanFactory",
    "TelemetryExporter",
    "TracerConfig",
]
