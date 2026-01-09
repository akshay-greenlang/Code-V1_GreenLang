# -*- coding: utf-8 -*-
"""
GL-017 CONDENSYNC - Observability Module
=========================================

Comprehensive observability infrastructure for condenser optimization.
Provides structured logging with JSON output and OpenTelemetry distributed tracing.

Components:
- logging.py: Structured JSON logging with correlation IDs and sensitive data redaction
- tracing.py: OpenTelemetry tracing with Jaeger/Zipkin/OTLP export

Features:
- JSON-formatted logs for ELK/Loki ingestion
- Correlation ID injection for request tracing
- Sensitive data redaction (GDPR compliant)
- OpenTelemetry-compatible distributed tracing
- Configurable sampling strategies
- Condenser-specific context propagation

Example:
    >>> from observability import (
    ...     StructuredLogger,
    ...     TracingManager,
    ...     traced,
    ...     CorrelationContext,
    ... )
    >>>
    >>> # Initialize logging
    >>> logger = StructuredLogger("gl-017-condensync")
    >>> logger.info("Processing condenser", condenser_id="COND-001", cf=0.85)
    >>>
    >>> # Initialize tracing
    >>> tracing = TracingManager(
    ...     service_name="gl-017-condensync",
    ...     exporter_type="jaeger",
    ... )
    >>>
    >>> # Use correlation context
    >>> with CorrelationContext("request-123", condenser_id="COND-001"):
    ...     logger.info("Processing with context")
    ...     with tracing.start_span("calculate_cf") as span:
    ...         result = calculate_cf(data)

Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "GreenLang Technologies"

# Logging exports
from .logging import (
    # Main classes
    StructuredLogger,
    BoundLogger,
    LogConfig,
    # Context managers
    CorrelationContext,
    CondenserLogContext,
    # Enums
    LogLevel,
    LogFormat,
    # Functions
    get_logger,
    configure_root_logger,
    set_correlation_id,
    get_correlation_id,
    set_condenser_context,
    get_condenser_context,
    add_log_context,
    clear_log_context,
    redact_sensitive_data,
    # Formatters
    JSONFormatter,
    TextFormatter,
    # Redactor
    SensitiveDataRedactor,
)

# Tracing exports
from .tracing import (
    # Main classes
    TracingManager,
    TracingConfig,
    Span,
    TraceContext,
    SpanAttributes,
    # Enums
    SpanKind,
    SamplingStrategy,
    ExporterType,
    CalculationSpanType,
    # Decorators
    traced,
    traced_async,
    traced_calculation,
    # Functions
    get_current_span,
    get_current_trace_id,
    inject_trace_context,
    extract_trace_context,
    # Exporters
    SpanExporter,
    ConsoleSpanExporter,
    JaegerExporter,
    ZipkinExporter,
    OTLPExporter,
    # Samplers
    Sampler,
    AlwaysOnSampler,
    AlwaysOffSampler,
    TraceIdRatioSampler,
    ParentBasedSampler,
)


__all__ = [
    # Version
    "__version__",

    # Logging - Main classes
    "StructuredLogger",
    "BoundLogger",
    "LogConfig",

    # Logging - Context managers
    "CorrelationContext",
    "CondenserLogContext",

    # Logging - Enums
    "LogLevel",
    "LogFormat",

    # Logging - Functions
    "get_logger",
    "configure_root_logger",
    "set_correlation_id",
    "get_correlation_id",
    "set_condenser_context",
    "get_condenser_context",
    "add_log_context",
    "clear_log_context",
    "redact_sensitive_data",

    # Logging - Formatters
    "JSONFormatter",
    "TextFormatter",

    # Logging - Redactor
    "SensitiveDataRedactor",

    # Tracing - Main classes
    "TracingManager",
    "TracingConfig",
    "Span",
    "TraceContext",
    "SpanAttributes",

    # Tracing - Enums
    "SpanKind",
    "SamplingStrategy",
    "ExporterType",
    "CalculationSpanType",

    # Tracing - Decorators
    "traced",
    "traced_async",
    "traced_calculation",

    # Tracing - Functions
    "get_current_span",
    "get_current_trace_id",
    "inject_trace_context",
    "extract_trace_context",

    # Tracing - Exporters
    "SpanExporter",
    "ConsoleSpanExporter",
    "JaegerExporter",
    "ZipkinExporter",
    "OTLPExporter",

    # Tracing - Samplers
    "Sampler",
    "AlwaysOnSampler",
    "AlwaysOffSampler",
    "TraceIdRatioSampler",
    "ParentBasedSampler",
]
