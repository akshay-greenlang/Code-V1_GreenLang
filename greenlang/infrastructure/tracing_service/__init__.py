# -*- coding: utf-8 -*-
"""
GreenLang Distributed Tracing SDK - OBS-003
============================================

Unified, production-hardened distributed tracing layer for GreenLang Climate
OS.  Replaces the deprecated Jaeger Thrift exporter and 17+ scattered
agent-specific tracers with a single OTLP-only SDK.

**Key Features:**

- OTLP-only export (gRPC) to OpenTelemetry Collector / Grafana Tempo.
- W3C TraceContext + W3C Baggage propagation.
- Custom ``GreenLangSampler`` with per-service-category rate overrides
  (compliance agents and security services always sampled at 100%).
- Auto-instrumentation for FastAPI, httpx, psycopg, Redis, Celery, requests.
- GreenLang semantic conventions (``gl.*`` namespace).
- Trace-derived Prometheus metrics bridge.
- Complete no-op fallback when the OTel SDK is not installed.

**Quick Start (FastAPI):**

    >>> from fastapi import FastAPI
    >>> from greenlang.infrastructure.tracing_service import configure_tracing
    >>>
    >>> app = FastAPI()
    >>> configure_tracing(app, service_name="api-service")

**Quick Start (batch worker / CLI):**

    >>> from greenlang.infrastructure.tracing_service import (
    ...     configure_tracing, get_tracer, shutdown_tracing,
    ... )
    >>> configure_tracing(service_name="batch-worker")
    >>> tracer = get_tracer(__name__)
    >>> with tracer.start_as_current_span("process_batch") as span:
    ...     span.set_attribute("batch.size", 1000)
    >>> shutdown_tracing()

**Decorators:**

    >>> from greenlang.infrastructure.tracing_service import (
    ...     trace_operation, trace_agent, trace_pipeline,
    ... )
    >>>
    >>> @trace_operation(name="calculate_emissions")
    ... async def calculate(data):
    ...     return data * 2.5
    >>>
    >>> @trace_agent(agent_type="carbon-calc")
    ... async def run_agent(input_data):
    ...     return process(input_data)

**Context Propagation:**

    >>> from greenlang.infrastructure.tracing_service import (
    ...     inject_trace_context, extract_trace_context,
    ...     get_current_trace_id, set_tenant_context,
    ... )
    >>> headers = {}
    >>> inject_trace_context(headers)

Author: GreenLang Platform Team
Date: February 2026
Status: Production Ready
Version: 1.0.0
"""

from __future__ import annotations

__version__ = "1.0.0"

# ---------------------------------------------------------------------------
# Setup (top-level entry point)
# ---------------------------------------------------------------------------

from greenlang.infrastructure.tracing_service.setup import (
    configure_tracing,
    shutdown_tracing,
    is_tracing_enabled,
    get_active_config,
)

# ---------------------------------------------------------------------------
# Provider (tracer access)
# ---------------------------------------------------------------------------

from greenlang.infrastructure.tracing_service.provider import (
    get_tracer,
    get_tracer_provider,
    get_current_span,
    shutdown,
    force_flush,
    OTEL_AVAILABLE,
)

# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------

from greenlang.infrastructure.tracing_service.decorators import (
    trace_operation,
    trace_agent,
    trace_pipeline,
)

# ---------------------------------------------------------------------------
# Context propagation
# ---------------------------------------------------------------------------

from greenlang.infrastructure.tracing_service.context import (
    inject_trace_context,
    extract_trace_context,
    get_current_trace_id,
    get_current_span_id,
    get_current_trace_context,
    set_tenant_context,
    get_tenant_context,
    set_user_context,
    get_user_context,
    set_correlation_id,
    get_correlation_id,
    set_baggage_item,
    get_baggage_item,
    get_all_baggage,
)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

from greenlang.infrastructure.tracing_service.middleware import (
    TracingMiddleware,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

from greenlang.infrastructure.tracing_service.config import (
    TracingConfig,
)

# ---------------------------------------------------------------------------
# Span enrichment (semantic conventions)
# ---------------------------------------------------------------------------

from greenlang.infrastructure.tracing_service.span_enrichment import (
    SpanEnricher,
    GL_TENANT_ID,
    GL_REQUEST_ID,
    GL_USER_ID,
    GL_CORRELATION_ID,
    GL_AGENT_TYPE,
    GL_AGENT_ID,
    GL_AGENT_KEY,
    GL_AGENT_VERSION,
    GL_AGENT_OPERATION,
    GL_PIPELINE_ID,
    GL_PIPELINE_NAME,
    GL_PIPELINE_STAGE,
    GL_PIPELINE_STEP,
    GL_EMISSION_SCOPE,
    GL_EMISSION_CATEGORY,
    GL_REGULATION,
    GL_FRAMEWORK,
    GL_DATA_SOURCE,
    GL_CALCULATION_TYPE,
    GL_CALCULATION_METHOD,
    GL_REPORTING_PERIOD,
    GL_ENVIRONMENT,
    GL_DEPLOYMENT_ID,
    GL_DATA_QUALITY_SCORE,
    GL_PROVENANCE_HASH,
)

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

from greenlang.infrastructure.tracing_service.sampling import (
    create_sampler,
    COMPLIANCE_SERVICES,
    SECURITY_SERVICES,
    INFRASTRUCTURE_SERVICES,
)

# ---------------------------------------------------------------------------
# Metrics bridge
# ---------------------------------------------------------------------------

from greenlang.infrastructure.tracing_service.metrics_bridge import (
    MetricsBridge,
    get_metrics_bridge,
)

# ---------------------------------------------------------------------------
# Instrumentors (usually not called directly)
# ---------------------------------------------------------------------------

from greenlang.infrastructure.tracing_service.instrumentors import (
    setup_instrumentors,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Version
    "__version__",
    # Setup
    "configure_tracing",
    "shutdown_tracing",
    "is_tracing_enabled",
    "get_active_config",
    # Provider
    "get_tracer",
    "get_tracer_provider",
    "get_current_span",
    "shutdown",
    "force_flush",
    "OTEL_AVAILABLE",
    # Decorators
    "trace_operation",
    "trace_agent",
    "trace_pipeline",
    # Context propagation
    "inject_trace_context",
    "extract_trace_context",
    "get_current_trace_id",
    "get_current_span_id",
    "get_current_trace_context",
    "set_tenant_context",
    "get_tenant_context",
    "set_user_context",
    "get_user_context",
    "set_correlation_id",
    "get_correlation_id",
    "set_baggage_item",
    "get_baggage_item",
    "get_all_baggage",
    # Middleware
    "TracingMiddleware",
    # Configuration
    "TracingConfig",
    # Span enrichment
    "SpanEnricher",
    "GL_TENANT_ID",
    "GL_REQUEST_ID",
    "GL_USER_ID",
    "GL_CORRELATION_ID",
    "GL_AGENT_TYPE",
    "GL_AGENT_ID",
    "GL_AGENT_KEY",
    "GL_AGENT_VERSION",
    "GL_AGENT_OPERATION",
    "GL_PIPELINE_ID",
    "GL_PIPELINE_NAME",
    "GL_PIPELINE_STAGE",
    "GL_PIPELINE_STEP",
    "GL_EMISSION_SCOPE",
    "GL_EMISSION_CATEGORY",
    "GL_REGULATION",
    "GL_FRAMEWORK",
    "GL_DATA_SOURCE",
    "GL_CALCULATION_TYPE",
    "GL_CALCULATION_METHOD",
    "GL_REPORTING_PERIOD",
    "GL_ENVIRONMENT",
    "GL_DEPLOYMENT_ID",
    "GL_DATA_QUALITY_SCORE",
    "GL_PROVENANCE_HASH",
    # Sampling
    "create_sampler",
    "COMPLIANCE_SERVICES",
    "SECURITY_SERVICES",
    "INFRASTRUCTURE_SERVICES",
    # Metrics bridge
    "MetricsBridge",
    "get_metrics_bridge",
    # Instrumentors
    "setup_instrumentors",
]
