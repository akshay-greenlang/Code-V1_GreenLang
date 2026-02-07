# -*- coding: utf-8 -*-
"""
TracerProvider Setup - OTLP-only exporter with resource detection (OBS-003)

Manages the global OpenTelemetry TracerProvider lifecycle:
1. Resource creation with GreenLang-specific attributes.
2. BatchSpanProcessor with OTLP gRPC exporter (primary).
3. Optional ConsoleSpanExporter for local debugging.
4. Custom GreenLangSampler via ParentBased wrapper.
5. Graceful degradation to no-op when the OTel SDK is not installed.

The module exposes a ``get_tracer()`` function that always returns a usable
tracer -- either a real OTel tracer or a lightweight no-op implementation
so that call sites never need to check for availability.

Example:
    >>> from greenlang.infrastructure.tracing_service.provider import (
    ...     setup_provider, get_tracer, shutdown,
    ... )
    >>> setup_provider(TracingConfig(service_name="api-service"))
    >>> tracer = get_tracer("my_module")
    >>> with tracer.start_as_current_span("work") as span:
    ...     span.set_attribute("key", "value")
    >>> shutdown()

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

from greenlang.infrastructure.tracing_service.config import TracingConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional OTel imports
# ---------------------------------------------------------------------------

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.trace import Status, StatusCode

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Module-level state (guarded by _lock)
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_provider: Optional[Any] = None  # TracerProvider | None
_config: Optional[TracingConfig] = None
_initialized: bool = False


# ---------------------------------------------------------------------------
# No-op fallback classes
# ---------------------------------------------------------------------------

class _NoOpSpan:
    """Lightweight span substitute when the OTel SDK is absent."""

    def set_attribute(self, key: str, value: Any) -> None:
        """No-op."""

    def set_status(self, *args: Any, **kwargs: Any) -> None:
        """No-op."""

    def add_event(
        self, name: str, attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """No-op."""

    def record_exception(
        self, exception: BaseException, **kwargs: Any
    ) -> None:
        """No-op."""

    def end(self, *args: Any) -> None:
        """No-op."""

    def is_recording(self) -> bool:
        """Always returns False."""
        return False

    def get_span_context(self) -> "_NoOpSpanContext":
        """Return a no-op span context."""
        return _NoOpSpanContext()

    def __enter__(self) -> "_NoOpSpan":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class _NoOpSpanContext:
    """Minimal span-context for the no-op path."""

    trace_id: int = 0
    span_id: int = 0
    is_valid: bool = False
    is_remote: bool = False


class _NoOpTracer:
    """Tracer substitute when OTel is unavailable.

    Every method mirrors the real ``trace.Tracer`` interface but returns
    ``_NoOpSpan`` instances so call-sites do not need conditional logic.
    """

    def start_span(
        self,
        name: str,
        context: Any = None,
        kind: Any = None,
        attributes: Optional[Dict[str, Any]] = None,
        links: Any = None,
        start_time: Any = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
    ) -> _NoOpSpan:
        """Return a no-op span."""
        return _NoOpSpan()

    @contextmanager
    def start_as_current_span(
        self,
        name: str,
        context: Any = None,
        kind: Any = None,
        attributes: Optional[Dict[str, Any]] = None,
        links: Any = None,
        start_time: Any = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        end_on_exit: bool = True,
    ) -> Generator[_NoOpSpan, None, None]:
        """Yield a no-op span as a context manager."""
        yield _NoOpSpan()


# ---------------------------------------------------------------------------
# Provider setup
# ---------------------------------------------------------------------------

def setup_provider(config: TracingConfig) -> None:
    """Initialise the global TracerProvider with OTLP export.

    Safe to call multiple times -- subsequent calls are silently ignored.
    Thread-safe via module-level lock.

    Args:
        config: Tracing configuration.
    """
    global _provider, _config, _initialized

    with _lock:
        if _initialized:
            logger.debug("TracerProvider already initialised; skipping")
            return

        _config = config

        if not OTEL_AVAILABLE:
            logger.info(
                "OpenTelemetry SDK not installed; tracing will use no-op fallback"
            )
            _initialized = True
            return

        if not config.enabled:
            logger.info("Tracing disabled via configuration (enabled=False)")
            _initialized = True
            return

        # -- Resource ----------------------------------------------------------
        resource_attrs: Dict[str, Any] = {
            SERVICE_NAME: config.service_name,
            SERVICE_VERSION: config.service_version,
            "deployment.environment": config.environment,
            "service.namespace": "greenlang",
            "service.platform": "greenlang-climate-os",
            "host.name": os.getenv("HOSTNAME", "unknown"),
        }
        # Kubernetes-aware attributes
        k8s_pod = os.getenv("HOSTNAME", "")
        k8s_ns = os.getenv("K8S_NAMESPACE", "")
        k8s_node = os.getenv("K8S_NODE_NAME", "")
        if k8s_pod:
            resource_attrs["k8s.pod.name"] = k8s_pod
        if k8s_ns:
            resource_attrs["k8s.namespace.name"] = k8s_ns
        if k8s_node:
            resource_attrs["k8s.node.name"] = k8s_node

        resource = Resource.create(resource_attrs)

        # -- Sampler -----------------------------------------------------------
        from greenlang.infrastructure.tracing_service.sampling import create_sampler

        sampler = create_sampler(config)

        # -- TracerProvider ----------------------------------------------------
        provider_kwargs: Dict[str, Any] = {"resource": resource}
        if sampler is not None:
            provider_kwargs["sampler"] = sampler

        _provider = TracerProvider(**provider_kwargs)

        # -- OTLP gRPC exporter (primary) -------------------------------------
        exporter_kwargs: Dict[str, Any] = {
            "endpoint": config.otlp_endpoint,
            "insecure": config.otlp_insecure,
            "timeout": config.otlp_timeout,
        }
        if config.otlp_headers:
            exporter_kwargs["headers"] = tuple(config.otlp_headers.items())

        otlp_exporter = OTLPSpanExporter(**exporter_kwargs)
        _provider.add_span_processor(
            BatchSpanProcessor(
                otlp_exporter,
                max_queue_size=config.batch_max_queue_size,
                max_export_batch_size=config.batch_max_export_batch_size,
                schedule_delay_millis=config.batch_schedule_delay_ms,
                export_timeout_millis=config.batch_export_timeout_ms,
            )
        )

        # -- Console exporter (debug) ------------------------------------------
        if config.console_exporter:
            _provider.add_span_processor(
                BatchSpanProcessor(ConsoleSpanExporter())
            )

        # -- Register globally -------------------------------------------------
        trace.set_tracer_provider(_provider)

        _initialized = True
        logger.info(
            "TracerProvider initialised: service=%s endpoint=%s sampling=%.2f",
            config.service_name,
            config.otlp_endpoint,
            config.sampling_rate,
        )


# ---------------------------------------------------------------------------
# Public accessors
# ---------------------------------------------------------------------------

def get_tracer_provider() -> Optional[Any]:
    """Return the active TracerProvider, or ``None`` if not yet initialised.

    Returns:
        The global TracerProvider or None.
    """
    return _provider


def get_tracer(name: str = __name__) -> Any:
    """Return a tracer instance for the given module name.

    Always returns a usable object -- either a real OTel ``Tracer`` when the
    SDK is available and initialised, or a ``_NoOpTracer`` that silently
    discards all spans.

    Args:
        name: Instrumentation scope name (typically ``__name__``).

    Returns:
        A ``trace.Tracer`` or ``_NoOpTracer``.
    """
    if OTEL_AVAILABLE and _initialized and _provider is not None:
        return trace.get_tracer(name)
    return _NoOpTracer()


def get_current_span() -> Any:
    """Return the current active span.

    Returns:
        The current OTel span or a ``_NoOpSpan``.
    """
    if OTEL_AVAILABLE:
        return trace.get_current_span()
    return _NoOpSpan()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def shutdown() -> None:
    """Gracefully flush pending spans and shut down the TracerProvider.

    Safe to call even when tracing was never initialised.
    """
    global _provider, _initialized

    with _lock:
        if _provider is not None:
            try:
                _provider.shutdown()
                logger.info("TracerProvider shut down successfully")
            except Exception as exc:
                logger.warning("Error shutting down TracerProvider: %s", exc)
            finally:
                _provider = None
        _initialized = False


def force_flush(timeout_millis: int = 30000) -> bool:
    """Force-flush all pending spans synchronously.

    Args:
        timeout_millis: Maximum time to wait for flush to complete.

    Returns:
        True if flush succeeded within the timeout, False otherwise.
    """
    if _provider is not None and hasattr(_provider, "force_flush"):
        try:
            return _provider.force_flush(timeout_millis)
        except Exception as exc:
            logger.warning("Error during force_flush: %s", exc)
    return False


__all__ = [
    "setup_provider",
    "get_tracer_provider",
    "get_tracer",
    "get_current_span",
    "shutdown",
    "force_flush",
    "OTEL_AVAILABLE",
    "_NoOpTracer",
    "_NoOpSpan",
]
