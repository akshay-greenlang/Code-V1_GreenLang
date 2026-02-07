# -*- coding: utf-8 -*-
"""
AgentTracer - OpenTelemetry tracer wrapper for GreenLang agent execution.

Provides a singleton OTLP tracer provider, root-span creation for agent
invocations, and trace-context injection/extraction for inter-agent
messaging.

Example:
    >>> tracer = AgentTracer.get_instance(TracerConfig(service_name="agent-factory"))
    >>> with tracer.start_span("process", agent_key="carbon-calc") as span:
    ...     span.set_attribute("input.records", 1000)

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Generator, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TracerConfig:
    """Configuration for the OTLP tracer provider.

    Attributes:
        service_name: Logical service name reported in traces.
        endpoint: OTLP collector endpoint.
        sampling_rate: Fraction of traces to sample (0.0-1.0).
        export_format: Export wire format (otlp, zipkin, jaeger).
        insecure: Use insecure gRPC channel (dev only).
        headers: Additional headers for the exporter.
    """

    service_name: str = "greenlang-agent-factory"
    endpoint: str = "http://localhost:4317"
    sampling_rate: float = 1.0
    export_format: str = "otlp"
    insecure: bool = True
    headers: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Lightweight span wrapper (avoids hard OTel dependency at import time)
# ---------------------------------------------------------------------------

class _SpanWrapper:
    """Thin wrapper around an OTel Span or a no-op fallback.

    Provides set_attribute, add_event, set_status, and end methods
    regardless of whether the opentelemetry SDK is installed.
    """

    def __init__(self, otel_span: Any = None, name: str = "") -> None:
        self._span = otel_span
        self.name = name
        self._attributes: Dict[str, Any] = {}
        self._events: list[Dict[str, Any]] = []

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self._attributes[key] = value
        if self._span is not None:
            try:
                self._span.set_attribute(key, value)
            except Exception:
                pass

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Record a span event."""
        event = {"name": name, "attributes": attributes or {}}
        self._events.append(event)
        if self._span is not None:
            try:
                self._span.add_event(name, attributes=attributes)
            except Exception:
                pass

    def set_status_ok(self) -> None:
        """Mark span as successful."""
        if self._span is not None:
            try:
                from opentelemetry.trace import StatusCode
                self._span.set_status(StatusCode.OK)
            except Exception:
                pass

    def set_status_error(self, description: str = "") -> None:
        """Mark span as errored."""
        if self._span is not None:
            try:
                from opentelemetry.trace import StatusCode
                self._span.set_status(StatusCode.ERROR, description)
            except Exception:
                pass

    def end(self) -> None:
        """End the span."""
        if self._span is not None:
            try:
                self._span.end()
            except Exception:
                pass

    def __enter__(self) -> "_SpanWrapper":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is not None:
            self.set_status_error(str(exc_val))
        else:
            self.set_status_ok()
        self.end()


# ---------------------------------------------------------------------------
# AgentTracer Singleton
# ---------------------------------------------------------------------------

class AgentTracer:
    """Singleton OpenTelemetry tracer for the Agent Factory.

    Wraps the OTel TracerProvider so the rest of the codebase does not
    need a hard dependency on the opentelemetry SDK.  When the SDK is
    not installed, spans degrade gracefully to no-ops.

    Attributes:
        config: Active TracerConfig.

    Example:
        >>> tracer = AgentTracer.get_instance()
        >>> with tracer.start_span("my-op") as span:
        ...     span.set_attribute("key", "value")
    """

    _instance: Optional["AgentTracer"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, config: TracerConfig) -> None:
        """Initialize the tracer (prefer get_instance for singleton access).

        Args:
            config: Tracer configuration.
        """
        self.config = config
        self._tracer: Any = None
        self._provider: Any = None
        self._setup_provider()

    # ---- Singleton -------------------------------------------------------

    @classmethod
    def get_instance(cls, config: Optional[TracerConfig] = None) -> "AgentTracer":
        """Return the singleton AgentTracer, creating it if necessary.

        Args:
            config: Optional configuration (used only on first call).

        Returns:
            The singleton AgentTracer instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(config or TracerConfig())
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (useful in tests)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.shutdown()
            cls._instance = None

    # ---- Provider setup --------------------------------------------------

    def _setup_provider(self) -> None:
        """Initialize the OTel TracerProvider if the SDK is available."""
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
            from opentelemetry.sdk.resources import Resource

            resource = Resource.create({
                "service.name": self.config.service_name,
                "service.version": "1.0.0",
                "deployment.environment": "production",
            })
            sampler = TraceIdRatioBased(self.config.sampling_rate)
            self._provider = TracerProvider(resource=resource, sampler=sampler)
            trace.set_tracer_provider(self._provider)
            self._tracer = trace.get_tracer(
                self.config.service_name, "1.0.0"
            )
            logger.info(
                "OTel TracerProvider initialized (endpoint=%s, sampling=%.2f)",
                self.config.endpoint,
                self.config.sampling_rate,
            )
        except ImportError:
            logger.info("opentelemetry SDK not installed; tracing disabled (no-op)")
            self._tracer = None

    # ---- Span creation ---------------------------------------------------

    @contextmanager
    def start_span(
        self,
        name: str,
        *,
        agent_key: str = "",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[_SpanWrapper, None, None]:
        """Create and yield a span wrapper.

        Args:
            name: Span operation name.
            agent_key: Agent key to tag the span with.
            attributes: Initial span attributes.

        Yields:
            A _SpanWrapper that supports set_attribute, add_event, etc.
        """
        attrs = dict(attributes or {})
        if agent_key:
            attrs["agent.key"] = agent_key

        if self._tracer is not None:
            otel_span = self._tracer.start_span(name, attributes=attrs)
            wrapper = _SpanWrapper(otel_span, name=name)
        else:
            wrapper = _SpanWrapper(name=name)
            for k, v in attrs.items():
                wrapper.set_attribute(k, v)

        try:
            yield wrapper
        except Exception as exc:
            wrapper.set_status_error(str(exc))
            raise
        finally:
            wrapper.set_status_ok()
            wrapper.end()

    def create_root_span(
        self,
        operation: str,
        agent_key: str,
        version: str,
        tenant_id: str = "",
    ) -> _SpanWrapper:
        """Create a root span for an agent invocation.

        Args:
            operation: Operation name (e.g. "execute", "deploy").
            agent_key: Agent identifier.
            version: Agent version.
            tenant_id: Optional tenant identifier.

        Returns:
            A _SpanWrapper.  Caller must call .end() when done.
        """
        attrs = {
            "agent.key": agent_key,
            "agent.version": version,
            "agent.operation": operation,
        }
        if tenant_id:
            attrs["tenant.id"] = tenant_id

        if self._tracer is not None:
            otel_span = self._tracer.start_span(
                f"agent.{operation}", attributes=attrs
            )
            return _SpanWrapper(otel_span, name=f"agent.{operation}")
        return _SpanWrapper(name=f"agent.{operation}")

    # ---- Context propagation ---------------------------------------------

    def inject_context(self, carrier: Dict[str, str]) -> Dict[str, str]:
        """Inject the current trace context into a carrier dict.

        Args:
            carrier: Mutable dict (e.g. HTTP headers) for context injection.

        Returns:
            The carrier with trace context headers added.
        """
        try:
            from opentelemetry.propagators import textmap
            from opentelemetry import context as otel_context

            propagator = textmap.get_global_textmap()
            propagator.inject(carrier)
        except ImportError:
            pass
        return carrier

    def extract_context(self, carrier: Dict[str, str]) -> Any:
        """Extract trace context from an incoming carrier dict.

        Args:
            carrier: Immutable dict with trace-context headers.

        Returns:
            OTel Context or None if extraction is not possible.
        """
        try:
            from opentelemetry.propagators import textmap

            propagator = textmap.get_global_textmap()
            return propagator.extract(carrier)
        except ImportError:
            return None

    # ---- Lifecycle -------------------------------------------------------

    def shutdown(self) -> None:
        """Flush pending spans and shut down the provider."""
        if self._provider is not None:
            try:
                self._provider.shutdown()
                logger.info("TracerProvider shut down")
            except Exception as exc:
                logger.warning("Error shutting down TracerProvider: %s", exc)


__all__ = ["AgentTracer", "TracerConfig", "_SpanWrapper"]
