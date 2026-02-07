# -*- coding: utf-8 -*-
"""
TelemetryExporter - Multi-backend span exporter for the GreenLang Agent Factory.

Supports OTLP (default), Jaeger, Zipkin, and console (dev) export backends.
Batches spans for efficiency and provides graceful shutdown with flush.

Example:
    >>> cfg = ExporterConfig(endpoint="http://otel-collector:4317", protocol="grpc")
    >>> exporter = TelemetryExporter(cfg)
    >>> exporter.start()
    >>> # ... spans are exported automatically via the tracer provider ...
    >>> exporter.shutdown()

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class ExportProtocol(str, Enum):
    """Wire protocol for span export."""

    GRPC = "grpc"
    HTTP = "http"


class ExportBackend(str, Enum):
    """Supported export destinations."""

    OTLP = "otlp"
    JAEGER = "jaeger"
    ZIPKIN = "zipkin"
    CONSOLE = "console"


@dataclass(frozen=True)
class ExporterConfig:
    """Configuration for the TelemetryExporter.

    Attributes:
        endpoint: Collector endpoint URL.
        protocol: Wire protocol (grpc or http).
        backend: Export backend type.
        headers: Extra headers sent with each export request.
        timeout_ms: Per-batch export timeout in milliseconds.
        batch_size: Maximum spans per batch.
        export_interval_ms: Interval between batch exports.
        max_queue_size: Upper bound on queued spans before dropping.
    """

    endpoint: str = "http://localhost:4317"
    protocol: ExportProtocol = ExportProtocol.GRPC
    backend: ExportBackend = ExportBackend.OTLP
    headers: Dict[str, str] = field(default_factory=dict)
    timeout_ms: int = 10_000
    batch_size: int = 512
    export_interval_ms: int = 5_000
    max_queue_size: int = 2048


# ---------------------------------------------------------------------------
# TelemetryExporter
# ---------------------------------------------------------------------------

class TelemetryExporter:
    """Multi-backend span exporter with batching and graceful shutdown.

    When the OpenTelemetry SDK is available, registers the appropriate
    SpanExporter on the global TracerProvider.  When the SDK is absent,
    operates as a lightweight in-memory console exporter (dev mode).

    Attributes:
        config: Active ExporterConfig.

    Example:
        >>> exporter = TelemetryExporter(ExporterConfig(backend=ExportBackend.CONSOLE))
        >>> exporter.start()
        >>> exporter.shutdown()
    """

    def __init__(self, config: Optional[ExporterConfig] = None) -> None:
        """Initialize the exporter.

        Args:
            config: Exporter configuration (uses defaults if None).
        """
        self.config = config or ExporterConfig()
        self._span_processor: Any = None
        self._exporter_impl: Any = None
        self._started = False
        self._lock = threading.Lock()
        # Fallback in-memory buffer when OTel is unavailable
        self._console_buffer: List[Dict[str, Any]] = []

    # ---- Lifecycle -------------------------------------------------------

    def start(self) -> None:
        """Initialize and register the span exporter.

        Safe to call multiple times; subsequent calls are no-ops.
        """
        with self._lock:
            if self._started:
                return
            self._started = True

        self._exporter_impl = self._create_exporter()
        if self._exporter_impl is not None:
            self._attach_to_provider(self._exporter_impl)
            logger.info(
                "TelemetryExporter started (backend=%s, endpoint=%s)",
                self.config.backend.value,
                self.config.endpoint,
            )
        else:
            logger.info(
                "TelemetryExporter started in console-fallback mode "
                "(OTel SDK not installed)"
            )

    def shutdown(self) -> None:
        """Flush pending spans and shut down the exporter."""
        with self._lock:
            if not self._started:
                return
            self._started = False

        if self._span_processor is not None:
            try:
                self._span_processor.shutdown()
                logger.info("Span processor shut down")
            except Exception as exc:
                logger.warning("Error shutting down span processor: %s", exc)

        if self._exporter_impl is not None:
            try:
                self._exporter_impl.shutdown()
            except Exception as exc:
                logger.warning("Error shutting down exporter: %s", exc)

        logger.info("TelemetryExporter shut down")

    def flush(self, timeout_ms: int = 5_000) -> bool:
        """Force-flush any buffered spans.

        Args:
            timeout_ms: Maximum time to wait for the flush.

        Returns:
            True if the flush completed within the timeout.
        """
        if self._span_processor is not None:
            try:
                return self._span_processor.force_flush(timeout_millis=timeout_ms)
            except Exception as exc:
                logger.warning("Flush failed: %s", exc)
                return False
        return True

    # ---- Console fallback ------------------------------------------------

    def export_span_dict(self, span_dict: Dict[str, Any]) -> None:
        """Export a span dict to the console buffer (fallback mode).

        This is used when the OTel SDK is not installed.

        Args:
            span_dict: Serialized span data.
        """
        if self.config.backend == ExportBackend.CONSOLE or self._exporter_impl is None:
            self._console_buffer.append(span_dict)
            if len(self._console_buffer) > self.config.max_queue_size:
                dropped = len(self._console_buffer) - self.config.max_queue_size
                self._console_buffer = self._console_buffer[-self.config.max_queue_size:]
                logger.warning("Console buffer full; dropped %d spans", dropped)
            logger.debug("Span exported to console buffer: %s", span_dict.get("span_name", ""))

    def get_console_buffer(self) -> List[Dict[str, Any]]:
        """Return and clear the console fallback buffer.

        Returns:
            List of span dictionaries.
        """
        buf = list(self._console_buffer)
        self._console_buffer.clear()
        return buf

    # ---- Internal --------------------------------------------------------

    def _create_exporter(self) -> Any:
        """Create the backend-specific span exporter."""
        backend = self.config.backend

        if backend == ExportBackend.CONSOLE:
            return self._create_console_exporter()

        if backend == ExportBackend.OTLP:
            return self._create_otlp_exporter()

        if backend == ExportBackend.JAEGER:
            return self._create_jaeger_exporter()

        if backend == ExportBackend.ZIPKIN:
            return self._create_zipkin_exporter()

        logger.warning("Unknown backend '%s'; falling back to console", backend)
        return self._create_console_exporter()

    def _create_otlp_exporter(self) -> Any:
        """Create an OTLP span exporter."""
        try:
            if self.config.protocol == ExportProtocol.GRPC:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )
            else:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                    OTLPSpanExporter,
                )

            return OTLPSpanExporter(
                endpoint=self.config.endpoint,
                headers=self.config.headers or None,
                timeout=self.config.timeout_ms / 1000,
            )
        except ImportError:
            logger.warning("OTLP exporter not installed; falling back to console")
            return self._create_console_exporter()

    def _create_jaeger_exporter(self) -> Any:
        """Create a Jaeger span exporter."""
        try:
            from opentelemetry.exporter.jaeger.thrift import JaegerExporter

            return JaegerExporter(
                agent_host_name=self.config.endpoint.split("://")[-1].split(":")[0],
                agent_port=int(self.config.endpoint.rsplit(":", 1)[-1])
                if ":" in self.config.endpoint.rsplit("/", 1)[-1]
                else 6831,
            )
        except ImportError:
            logger.warning("Jaeger exporter not installed; falling back to console")
            return self._create_console_exporter()

    def _create_zipkin_exporter(self) -> Any:
        """Create a Zipkin span exporter."""
        try:
            from opentelemetry.exporter.zipkin.json import ZipkinExporter

            return ZipkinExporter(endpoint=self.config.endpoint)
        except ImportError:
            logger.warning("Zipkin exporter not installed; falling back to console")
            return self._create_console_exporter()

    def _create_console_exporter(self) -> Any:
        """Create a console span exporter (always available)."""
        try:
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter
            return ConsoleSpanExporter()
        except ImportError:
            return None

    def _attach_to_provider(self, exporter: Any) -> None:
        """Attach a span exporter to the global TracerProvider."""
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            provider = trace.get_tracer_provider()
            if not isinstance(provider, TracerProvider):
                logger.warning("Global TracerProvider is not an SDK provider; cannot attach exporter")
                return

            self._span_processor = BatchSpanProcessor(
                exporter,
                max_queue_size=self.config.max_queue_size,
                max_export_batch_size=self.config.batch_size,
                schedule_delay_millis=self.config.export_interval_ms,
                export_timeout_millis=self.config.timeout_ms,
            )
            provider.add_span_processor(self._span_processor)
            logger.info("BatchSpanProcessor attached to TracerProvider")
        except ImportError:
            logger.info("OTel SDK not available; exporter not attached")
        except Exception as exc:
            logger.warning("Failed to attach exporter to provider: %s", exc)


__all__ = ["ExportBackend", "ExporterConfig", "ExportProtocol", "TelemetryExporter"]
