# -*- coding: utf-8 -*-
"""
GreenLang Agent Factory - Prometheus Exporters

This module provides Prometheus metric exporters and HTTP server functionality
for exposing metrics from GreenLang agents.

Features:
- Standalone metrics HTTP server
- FastAPI integration helpers
- Custom registry support
- Multi-process mode support

Created: 2025-12-03
Team: Monitoring & Observability
Version: 1.0.0
"""

import logging
import threading
from typing import Optional, Dict, Any
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver

from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
    multiprocess,
    start_http_server,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Prometheus Exporter
# ============================================================================

class PrometheusExporter:
    """
    Prometheus metrics exporter for GreenLang agents.

    This class provides a standalone HTTP server for exposing Prometheus
    metrics. It can be used when the agent doesn't have a built-in HTTP
    server (e.g., CLI tools, background workers).

    Attributes:
        port: HTTP port for metrics endpoint
        host: HTTP host for metrics endpoint
        registry: Prometheus CollectorRegistry to use

    Example:
        >>> exporter = PrometheusExporter(port=9090)
        >>> exporter.start()
        >>> # ... your agent logic ...
        >>> exporter.stop()
    """

    def __init__(
        self,
        port: int = 9090,
        host: str = "0.0.0.0",
        registry: Optional[CollectorRegistry] = None,
    ):
        """
        Initialize the PrometheusExporter.

        Args:
            port: Port to expose metrics on
            host: Host to bind to
            registry: Optional custom registry (defaults to REGISTRY)
        """
        self.port = port
        self.host = host
        self.registry = registry or REGISTRY
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """
        Start the metrics HTTP server in a background thread.

        This method is non-blocking and returns immediately.
        """
        if self._server is not None:
            logger.warning("Metrics server already running")
            return

        try:
            start_http_server(
                port=self.port,
                addr=self.host,
                registry=self.registry,
            )
            logger.info(f"Prometheus metrics server started on {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            raise

    def stop(self) -> None:
        """
        Stop the metrics HTTP server.

        Note: The prometheus_client start_http_server doesn't provide
        a clean shutdown mechanism. Consider using create_metrics_app
        for production deployments.
        """
        logger.info("Metrics server shutdown requested")
        # Note: prometheus_client.start_http_server doesn't expose server handle
        # For clean shutdown, use the custom server or FastAPI integration


class MetricsHTTPHandler(BaseHTTPRequestHandler):
    """HTTP request handler for Prometheus metrics."""

    registry: CollectorRegistry = REGISTRY

    def do_GET(self):
        """Handle GET requests to /metrics endpoint."""
        if self.path == "/metrics":
            try:
                output = generate_latest(self.registry)
                self.send_response(200)
                self.send_header("Content-Type", CONTENT_TYPE_LATEST)
                self.end_headers()
                self.wfile.write(output)
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(f"Error generating metrics: {e}".encode())
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "healthy"}')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress default access logging."""
        pass


class ThreadedMetricsServer(socketserver.ThreadingMixIn, HTTPServer):
    """Threaded HTTP server for metrics."""
    allow_reuse_address = True
    daemon_threads = True


def create_custom_exporter(
    port: int = 9090,
    host: str = "0.0.0.0",
    registry: Optional[CollectorRegistry] = None,
) -> ThreadedMetricsServer:
    """
    Create a custom threaded metrics server.

    This provides more control over the server lifecycle than
    prometheus_client.start_http_server.

    Args:
        port: Port to bind to
        host: Host to bind to
        registry: Optional custom registry

    Returns:
        ThreadedMetricsServer instance

    Example:
        >>> server = create_custom_exporter(port=9090)
        >>> thread = threading.Thread(target=server.serve_forever)
        >>> thread.daemon = True
        >>> thread.start()
        >>> # ... later ...
        >>> server.shutdown()
    """
    if registry is not None:
        MetricsHTTPHandler.registry = registry

    server = ThreadedMetricsServer((host, port), MetricsHTTPHandler)
    return server


# ============================================================================
# FastAPI Integration
# ============================================================================

def create_metrics_app(
    registry: Optional[CollectorRegistry] = None,
) -> Dict[str, Any]:
    """
    Create a metrics endpoint configuration for FastAPI.

    This function returns a dict that can be passed to app.add_api_route().

    Args:
        registry: Optional custom registry

    Returns:
        Dict with route configuration

    Example:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> metrics_route = create_metrics_app()
        >>> app.add_api_route(**metrics_route)
    """
    from fastapi import Response

    _registry = registry or REGISTRY

    async def metrics_endpoint():
        """Prometheus metrics endpoint."""
        return Response(
            content=generate_latest(_registry),
            media_type=CONTENT_TYPE_LATEST,
        )

    return {
        "path": "/metrics",
        "endpoint": metrics_endpoint,
        "methods": ["GET"],
        "tags": ["Monitoring"],
        "name": "prometheus_metrics",
        "summary": "Prometheus metrics endpoint",
        "description": "Returns Prometheus-formatted metrics for monitoring",
    }


def add_metrics_to_fastapi(app: Any, registry: Optional[CollectorRegistry] = None) -> None:
    """
    Add metrics endpoint to a FastAPI application.

    Args:
        app: FastAPI application instance
        registry: Optional custom registry
    """
    from fastapi import Response

    _registry = registry or REGISTRY

    @app.get("/metrics", tags=["Monitoring"])
    async def metrics():
        """Prometheus metrics endpoint."""
        return Response(
            content=generate_latest(_registry),
            media_type=CONTENT_TYPE_LATEST,
        )

    @app.get("/metrics/health", tags=["Monitoring"])
    async def metrics_health():
        """Metrics endpoint health check."""
        return {"status": "healthy", "metrics_enabled": True}


# ============================================================================
# Multiprocess Mode Support
# ============================================================================

def create_multiprocess_registry(prometheus_multiproc_dir: str) -> CollectorRegistry:
    """
    Create a registry for multiprocess mode (e.g., Gunicorn workers).

    This is required when running with multiple worker processes to
    properly aggregate metrics across all workers.

    Args:
        prometheus_multiproc_dir: Directory for sharing metrics between processes

    Returns:
        CollectorRegistry configured for multiprocess mode

    Example:
        >>> import os
        >>> os.environ["prometheus_multiproc_dir"] = "/tmp/prometheus_multiproc"
        >>> registry = create_multiprocess_registry("/tmp/prometheus_multiproc")
    """
    import os
    os.makedirs(prometheus_multiproc_dir, exist_ok=True)
    os.environ["prometheus_multiproc_dir"] = prometheus_multiproc_dir

    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    return registry


def cleanup_multiprocess_registry(prometheus_multiproc_dir: str) -> None:
    """
    Cleanup multiprocess registry files on shutdown.

    Call this in your atexit handler or shutdown hook.

    Args:
        prometheus_multiproc_dir: Directory containing multiprocess files
    """
    import shutil
    try:
        shutil.rmtree(prometheus_multiproc_dir, ignore_errors=True)
        logger.info(f"Cleaned up multiprocess registry: {prometheus_multiproc_dir}")
    except Exception as e:
        logger.warning(f"Failed to cleanup multiprocess registry: {e}")
