# -*- coding: utf-8 -*-
"""
GL-010 EmissionsGuardian - Health Check Endpoints Module

This module provides comprehensive health check endpoints for Kubernetes
probes and monitoring systems. Includes liveness, readiness, and startup
probes with component health aggregation.

Endpoints:
    - /health/live - Liveness probe (is the process alive?)
    - /health/ready - Readiness probe (are all dependencies up?)
    - /health/startup - Startup probe (has initialization completed?)

Standards Compliance:
    - Kubernetes health check patterns
    - GreenLang health monitoring standards
    - JSON response format for observability

Example:
    >>> from monitoring.health import HealthCheckServer
    >>> server = HealthCheckServer(port=8010)
    >>> server.start()
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import socket
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Callable, Dict, List, Optional, Set
from urllib.parse import urlparse
import json

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

HEALTH_CHECK_PORT = 8010
DEFAULT_TIMEOUT_SECONDS = 5.0
DEFAULT_CHECK_INTERVAL_SECONDS = 30.0


# =============================================================================
# Health Status Enums
# =============================================================================

class HealthStatus(str, Enum):
    """Health status values."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentType(str, Enum):
    """Component types for health checks."""
    DATABASE = "database"
    CACHE = "cache"
    MESSAGE_QUEUE = "message_queue"
    EXTERNAL_API = "external_api"
    FILE_SYSTEM = "file_system"
    DAHS = "dahs"
    SCADA = "scada"
    CEMS = "cems"
    ERP = "erp"
    INTERNAL = "internal"


# =============================================================================
# Health Check Models
# =============================================================================

@dataclass
class ComponentHealth:
    """Health status for a single component."""
    name: str
    component_type: ComponentType
    status: HealthStatus
    latency_ms: float = 0.0
    message: Optional[str] = None
    last_check: datetime = field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "type": self.component_type.value,
            "status": self.status.value,
            "latency_ms": round(self.latency_ms, 2),
            "message": self.message,
            "last_check": self.last_check.isoformat(),
            "details": self.details
        }


@dataclass
class HealthCheckResult:
    """Aggregated health check result."""
    status: HealthStatus
    components: List[ComponentHealth]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    version: str = "1.0.0"
    uptime_seconds: float = 0.0
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "provenance_hash": self.provenance_hash,
            "components": [c.to_dict() for c in self.components]
        }

    def calculate_provenance(self) -> str:
        """Calculate SHA-256 hash of health check result."""
        content = f"{self.status.value}|{self.timestamp.isoformat()}|{len(self.components)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]



# =============================================================================
# Health Check Functions
# =============================================================================

class HealthChecker:
    """Base class for health check implementations."""

    def __init__(self, name: str, component_type: ComponentType, timeout: float = DEFAULT_TIMEOUT_SECONDS):
        self.name = name
        self.component_type = component_type
        self.timeout = timeout
        self._last_result: Optional[ComponentHealth] = None

    def check(self) -> ComponentHealth:
        """Perform health check and return result."""
        raise NotImplementedError

    @property
    def last_result(self) -> Optional[ComponentHealth]:
        return self._last_result


class TCPHealthChecker(HealthChecker):
    """TCP connection health checker."""

    def __init__(self, name: str, component_type: ComponentType, host: str, port: int, timeout: float = DEFAULT_TIMEOUT_SECONDS):
        super().__init__(name, component_type, timeout)
        self.host = host
        self.port = port

    def check(self) -> ComponentHealth:
        start_time = time.perf_counter()
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect((self.host, self.port))
            sock.close()
            latency = (time.perf_counter() - start_time) * 1000
            self._last_result = ComponentHealth(
                name=self.name,
                component_type=self.component_type,
                status=HealthStatus.HEALTHY,
                latency_ms=latency,
                message=f"Connected to {self.host}:{self.port}",
                details={"host": self.host, "port": self.port}
            )
        except socket.timeout:
            latency = (time.perf_counter() - start_time) * 1000
            self._last_result = ComponentHealth(
                name=self.name,
                component_type=self.component_type,
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency,
                message=f"Connection timeout to {self.host}:{self.port}",
                details={"host": self.host, "port": self.port, "error": "timeout"}
            )
        except Exception as e:
            latency = (time.perf_counter() - start_time) * 1000
            self._last_result = ComponentHealth(
                name=self.name,
                component_type=self.component_type,
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency,
                message=str(e),
                details={"host": self.host, "port": self.port, "error": str(e)}
            )
        return self._last_result


class CallableHealthChecker(HealthChecker):
    """Health checker using a callable function."""

    def __init__(self, name: str, component_type: ComponentType, check_fn: Callable[[], bool], timeout: float = DEFAULT_TIMEOUT_SECONDS):
        super().__init__(name, component_type, timeout)
        self.check_fn = check_fn

    def check(self) -> ComponentHealth:
        start_time = time.perf_counter()
        try:
            result = self.check_fn()
            latency = (time.perf_counter() - start_time) * 1000
            status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
            self._last_result = ComponentHealth(
                name=self.name,
                component_type=self.component_type,
                status=status,
                latency_ms=latency,
                message="Check passed" if result else "Check failed"
            )
        except Exception as e:
            latency = (time.perf_counter() - start_time) * 1000
            self._last_result = ComponentHealth(
                name=self.name,
                component_type=self.component_type,
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency,
                message=str(e),
                details={"error": str(e)}
            )
        return self._last_result



# =============================================================================
# Health Check Registry
# =============================================================================

class HealthCheckRegistry:
    """Registry for managing health checks."""

    _instance: Optional['HealthCheckRegistry'] = None
    _lock = threading.Lock()

    def __new__(cls) -> 'HealthCheckRegistry':
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if getattr(self, '_initialized', False):
            return
        self._initialized = True
        self._startup_time = datetime.utcnow()
        self._startup_complete = False
        self._liveness_checks: Dict[str, HealthChecker] = {}
        self._readiness_checks: Dict[str, HealthChecker] = {}
        self._last_check_time: Optional[datetime] = None
        self._check_lock = threading.Lock()
        logger.info('HealthCheckRegistry initialized')

    def register_liveness_check(self, checker: HealthChecker) -> None:
        """Register a liveness check."""
        with self._check_lock:
            self._liveness_checks[checker.name] = checker
            logger.info(f'Registered liveness check: {checker.name}')

    def register_readiness_check(self, checker: HealthChecker) -> None:
        """Register a readiness check."""
        with self._check_lock:
            self._readiness_checks[checker.name] = checker
            logger.info(f'Registered readiness check: {checker.name}')

    def unregister_check(self, name: str) -> None:
        """Unregister a health check by name."""
        with self._check_lock:
            self._liveness_checks.pop(name, None)
            self._readiness_checks.pop(name, None)

    def mark_startup_complete(self) -> None:
        """Mark startup as complete."""
        self._startup_complete = True
        logger.info('Startup marked as complete')

    def get_uptime_seconds(self) -> float:
        """Get uptime in seconds."""
        return (datetime.utcnow() - self._startup_time).total_seconds()

    def check_liveness(self) -> HealthCheckResult:
        """Perform liveness check."""
        components = []
        overall_status = HealthStatus.HEALTHY

        # Basic process check - if we can execute this, we are alive
        components.append(ComponentHealth(
            name='process',
            component_type=ComponentType.INTERNAL,
            status=HealthStatus.HEALTHY,
            message='Process is running'
        ))

        with self._check_lock:
            for checker in self._liveness_checks.values():
                try:
                    result = checker.check()
                    components.append(result)
                    if result.status == HealthStatus.UNHEALTHY:
                        overall_status = HealthStatus.UNHEALTHY
                    elif result.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                        overall_status = HealthStatus.DEGRADED
                except Exception as e:
                    logger.error(f'Liveness check failed for {checker.name}: {e}')
                    components.append(ComponentHealth(
                        name=checker.name,
                        component_type=checker.component_type,
                        status=HealthStatus.UNHEALTHY,
                        message=str(e)
                    ))
                    overall_status = HealthStatus.UNHEALTHY

        result = HealthCheckResult(
            status=overall_status,
            components=components,
            uptime_seconds=self.get_uptime_seconds()
        )
        result.provenance_hash = result.calculate_provenance()
        return result

    def check_readiness(self) -> HealthCheckResult:
        """Perform readiness check."""
        components = []
        overall_status = HealthStatus.HEALTHY

        with self._check_lock:
            for checker in self._readiness_checks.values():
                try:
                    result = checker.check()
                    components.append(result)
                    if result.status == HealthStatus.UNHEALTHY:
                        overall_status = HealthStatus.UNHEALTHY
                    elif result.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                        overall_status = HealthStatus.DEGRADED
                except Exception as e:
                    logger.error(f'Readiness check failed for {checker.name}: {e}')
                    components.append(ComponentHealth(
                        name=checker.name,
                        component_type=checker.component_type,
                        status=HealthStatus.UNHEALTHY,
                        message=str(e)
                    ))
                    overall_status = HealthStatus.UNHEALTHY

        self._last_check_time = datetime.utcnow()
        result = HealthCheckResult(
            status=overall_status,
            components=components,
            uptime_seconds=self.get_uptime_seconds()
        )
        result.provenance_hash = result.calculate_provenance()
        return result

    def check_startup(self) -> HealthCheckResult:
        """Perform startup check."""
        if self._startup_complete:
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                components=[ComponentHealth(
                    name='startup',
                    component_type=ComponentType.INTERNAL,
                    status=HealthStatus.HEALTHY,
                    message='Startup complete'
                )],
                uptime_seconds=self.get_uptime_seconds()
            )
        else:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                components=[ComponentHealth(
                    name='startup',
                    component_type=ComponentType.INTERNAL,
                    status=HealthStatus.UNHEALTHY,
                    message='Startup in progress'
                )],
                uptime_seconds=self.get_uptime_seconds()
            )



# =============================================================================
# HTTP Health Check Server
# =============================================================================

class HealthHandler(BaseHTTPRequestHandler):
    """HTTP handler for health check endpoints."""
    registry: Optional[HealthCheckRegistry] = None

    def do_GET(self) -> None:
        if self.path == '/health/live':
            self._handle_liveness()
        elif self.path == '/health/ready':
            self._handle_readiness()
        elif self.path == '/health/startup':
            self._handle_startup()
        elif self.path == '/health':
            self._handle_all()
        else:
            self.send_error(404, 'Not Found')

    def _send_json_response(self, status_code: int, data: Dict[str, Any]) -> None:
        content = json.dumps(data, indent=2)
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(content)))
        self.end_headers()
        self.wfile.write(content.encode('utf-8'))

    def _handle_liveness(self) -> None:
        if self.registry is None:
            self.registry = HealthCheckRegistry()
        result = self.registry.check_liveness()
        status_code = 200 if result.status == HealthStatus.HEALTHY else 503
        self._send_json_response(status_code, result.to_dict())

    def _handle_readiness(self) -> None:
        if self.registry is None:
            self.registry = HealthCheckRegistry()
        result = self.registry.check_readiness()
        status_code = 200 if result.status == HealthStatus.HEALTHY else 503
        self._send_json_response(status_code, result.to_dict())

    def _handle_startup(self) -> None:
        if self.registry is None:
            self.registry = HealthCheckRegistry()
        result = self.registry.check_startup()
        status_code = 200 if result.status == HealthStatus.HEALTHY else 503
        self._send_json_response(status_code, result.to_dict())

    def _handle_all(self) -> None:
        if self.registry is None:
            self.registry = HealthCheckRegistry()
        liveness = self.registry.check_liveness()
        readiness = self.registry.check_readiness()
        startup = self.registry.check_startup()
        combined = {
            'liveness': liveness.to_dict(),
            'readiness': readiness.to_dict(),
            'startup': startup.to_dict()
        }
        overall_healthy = all([
            liveness.status == HealthStatus.HEALTHY,
            readiness.status == HealthStatus.HEALTHY,
            startup.status == HealthStatus.HEALTHY
        ])
        status_code = 200 if overall_healthy else 503
        self._send_json_response(status_code, combined)

    def log_message(self, format: str, *args: Any) -> None:
        logger.debug(f'Health HTTP: {format % args}')


class HealthCheckServer:
    """HTTP server for health check endpoints."""

    def __init__(self, host: str = '0.0.0.0', port: int = HEALTH_CHECK_PORT, registry: Optional[HealthCheckRegistry] = None):
        self.host = host
        self.port = port
        self.registry = registry or HealthCheckRegistry()
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        HealthHandler.registry = self.registry
        self._server = HTTPServer((self.host, self.port), HealthHandler)
        self._running = True
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        logger.info(f'Health check server started on http://{self.host}:{self.port}/health')

    def _serve(self) -> None:
        if self._server:
            self._server.serve_forever()

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._server:
            self._server.shutdown()
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info('Health check server stopped')

    def is_running(self) -> bool:
        return self._running


# =============================================================================
# Default Health Checkers for EmissionsGuardian Dependencies
# =============================================================================

def create_dahs_health_checker(host: str, port: int) -> TCPHealthChecker:
    """Create health checker for DAHS connection."""
    return TCPHealthChecker(
        name='dahs',
        component_type=ComponentType.DAHS,
        host=host,
        port=port,
        timeout=5.0
    )


def create_database_health_checker(host: str, port: int) -> TCPHealthChecker:
    """Create health checker for database connection."""
    return TCPHealthChecker(
        name='database',
        component_type=ComponentType.DATABASE,
        host=host,
        port=port,
        timeout=5.0
    )


def create_cems_health_checker(host: str, port: int) -> TCPHealthChecker:
    """Create health checker for CEMS connection."""
    return TCPHealthChecker(
        name='cems',
        component_type=ComponentType.CEMS,
        host=host,
        port=port,
        timeout=5.0
    )


# =============================================================================
# Module-level Convenience Functions
# =============================================================================

_default_registry: Optional[HealthCheckRegistry] = None
_default_server: Optional[HealthCheckServer] = None


def get_health_registry() -> HealthCheckRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = HealthCheckRegistry()
    return _default_registry


def start_health_server(host: str = '0.0.0.0', port: int = HEALTH_CHECK_PORT) -> HealthCheckServer:
    global _default_server
    if _default_server is None or not _default_server.is_running():
        _default_server = HealthCheckServer(host=host, port=port)
        _default_server.start()
    return _default_server


def stop_health_server() -> None:
    global _default_server
    if _default_server is not None:
        _default_server.stop()
        _default_server = None
