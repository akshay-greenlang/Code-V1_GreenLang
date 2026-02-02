# -*- coding: utf-8 -*-
"""
GreenLang Health Check System
============================

Production-ready health check endpoints for GreenLang v0.2.0.

This module provides:
- Component health checking
- HTTP health endpoints
- Dependency validation
- Performance health metrics
- Integration with monitoring systems

Usage:
    # Create health checker
    health = HealthChecker()

    # Add component checks
    health.add_component_check("database", check_database_connection)
    health.add_component_check("runtime", check_runtime_status)

    # Create health check web app
    app = create_health_app(health, port=8080)
    app.run()
"""

import asyncio
import json
import logging
import os
import psutil
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from uuid import uuid4

# HTTP server imports (try different options)
try:
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from urllib.parse import urlparse, parse_qs
    HAS_HTTP_SERVER = True
except ImportError:
    HAS_HTTP_SERVER = False

# Try to import optional web framework
try:
    from flask import Flask, jsonify, request
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False
    # Dummy Flask class
    class Flask:
        def __init__(self, *args, **kwargs):
            pass

try:
    from fastapi import FastAPI, Response
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    # Dummy FastAPI classes
    class FastAPI:
        def __init__(self, *args, **kwargs):
            pass

    class Response:
        def __init__(self, *args, **kwargs):
            pass

from .metrics import get_metrics

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status of a single component"""
    name: str
    status: HealthStatus
    message: str = ""
    details: Dict[str, Any] = None
    last_checked: float = None
    response_time_ms: Optional[float] = None
    error: Optional[str] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}
        if self.last_checked is None:
            self.last_checked = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['status'] = self.status.value
        data['last_checked_iso'] = datetime.fromtimestamp(self.last_checked).isoformat()
        return data


@dataclass
class SystemHealth:
    """Overall system health status"""
    status: HealthStatus
    components: Dict[str, ComponentHealth]
    timestamp: float
    version: str
    uptime_seconds: float
    summary: Dict[str, Any] = None

    def __post_init__(self):
        if self.summary is None:
            self.summary = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp,
            "timestamp_iso": datetime.fromtimestamp(self.timestamp).isoformat(),
            "version": self.version,
            "uptime_seconds": self.uptime_seconds,
            "components": {name: comp.to_dict() for name, comp in self.components.items()},
            "summary": self.summary
        }


class HealthChecker:
    """Main health checking system"""

    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.component_checks: Dict[str, Callable] = {}
        self.component_health: Dict[str, ComponentHealth] = {}
        self.start_time = time.time()
        self.last_full_check = 0
        self.version = "0.2.0"  # GreenLang version

        # Built-in checks
        self._setup_builtin_checks()

        logger.info("HealthChecker initialized")

    def _setup_builtin_checks(self):
        """Setup built-in system health checks"""
        self.add_component_check("system_resources", self._check_system_resources)
        self.add_component_check("python_runtime", self._check_python_runtime)
        self.add_component_check("greenlang_core", self._check_greenlang_core)
        self.add_component_check("metrics_collector", self._check_metrics_collector)

    def add_component_check(self, name: str, check_func: Callable[[], Union[ComponentHealth, Tuple[HealthStatus, str, Dict]]]):
        """Add a component health check function"""
        self.component_checks[name] = check_func
        logger.info(f"Added health check for component: {name}")

    def remove_component_check(self, name: str):
        """Remove a component health check"""
        if name in self.component_checks:
            del self.component_checks[name]
            if name in self.component_health:
                del self.component_health[name]
            logger.info(f"Removed health check for component: {name}")

    async def check_component(self, name: str) -> ComponentHealth:
        """Check health of a single component"""
        if name not in self.component_checks:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNKNOWN,
                message="No health check defined",
                error="Component not found"
            )

        check_func = self.component_checks[name]
        start_time = time.perf_counter()

        try:
            # Call the check function
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()

            response_time_ms = (time.perf_counter() - start_time) * 1000

            # Handle different return types
            if isinstance(result, ComponentHealth):
                result.response_time_ms = response_time_ms
                return result
            elif isinstance(result, tuple) and len(result) >= 2:
                status, message = result[:2]
                details = result[2] if len(result) > 2 else {}
                return ComponentHealth(
                    name=name,
                    status=status,
                    message=message,
                    details=details,
                    response_time_ms=response_time_ms
                )
            else:
                # Assume boolean result
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                return ComponentHealth(
                    name=name,
                    status=status,
                    message="Check completed",
                    response_time_ms=response_time_ms
                )

        except Exception as e:
            response_time_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Health check failed for {name}: {e}")

            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                response_time_ms=response_time_ms,
                error=str(e)
            )

    async def check_all_components(self) -> Dict[str, ComponentHealth]:
        """Check health of all registered components"""
        results = {}

        # Run all checks concurrently
        tasks = []
        for name in self.component_checks:
            tasks.append(self.check_component(name))

        component_results = await asyncio.gather(*tasks, return_exceptions=True)

        for name, result in zip(self.component_checks.keys(), component_results):
            if isinstance(result, Exception):
                results[name] = ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed with exception: {str(result)}",
                    error=str(result)
                )
            else:
                results[name] = result

        self.component_health = results
        self.last_full_check = time.time()
        return results

    async def get_system_health(self, use_cached: bool = True) -> SystemHealth:
        """Get overall system health status"""
        # Use cached results if they're recent enough
        if use_cached and self.component_health and (time.time() - self.last_full_check) < self.check_interval:
            components = self.component_health
        else:
            components = await self.check_all_components()

        # Determine overall status
        if not components:
            overall_status = HealthStatus.UNKNOWN
        else:
            statuses = [comp.status for comp in components.values()]
            if any(status == HealthStatus.UNHEALTHY for status in statuses):
                overall_status = HealthStatus.UNHEALTHY
            elif any(status == HealthStatus.DEGRADED for status in statuses):
                overall_status = HealthStatus.DEGRADED
            elif any(status == HealthStatus.UNKNOWN for status in statuses):
                overall_status = HealthStatus.DEGRADED
            else:
                overall_status = HealthStatus.HEALTHY

        # Generate summary
        summary = {
            "total_components": len(components),
            "healthy_components": len([c for c in components.values() if c.status == HealthStatus.HEALTHY]),
            "degraded_components": len([c for c in components.values() if c.status == HealthStatus.DEGRADED]),
            "unhealthy_components": len([c for c in components.values() if c.status == HealthStatus.UNHEALTHY]),
            "unknown_components": len([c for c in components.values() if c.status == HealthStatus.UNKNOWN]),
            "avg_response_time_ms": sum(c.response_time_ms or 0 for c in components.values()) / len(components) if components else 0
        }

        return SystemHealth(
            status=overall_status,
            components=components,
            timestamp=time.time(),
            version=self.version,
            uptime_seconds=time.time() - self.start_time,
            summary=summary
        )

    # Built-in health check functions
    def _check_system_resources(self) -> ComponentHealth:
        """Check system resource usage"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            cpu_percent = process.cpu_percent()

            # System-wide metrics
            vm = psutil.virtual_memory()
            disk_usage = psutil.disk_usage('/')

            # Define thresholds
            memory_threshold = 0.8  # 80% memory usage
            cpu_threshold = 80.0    # 80% CPU usage
            disk_threshold = 0.9    # 90% disk usage

            # Check for issues
            issues = []
            if vm.percent / 100 > memory_threshold:
                issues.append(f"High memory usage: {vm.percent:.1f}%")

            if cpu_percent > cpu_threshold:
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")

            if disk_usage.percent / 100 > disk_threshold:
                issues.append(f"High disk usage: {disk_usage.percent:.1f}%")

            # Determine status
            if len(issues) >= 2:
                status = HealthStatus.UNHEALTHY
                message = f"Multiple resource issues: {'; '.join(issues)}"
            elif issues:
                status = HealthStatus.DEGRADED
                message = issues[0]
            else:
                status = HealthStatus.HEALTHY
                message = "System resources are healthy"

            details = {
                "process_memory_mb": memory_info.rss / 1024 / 1024,
                "process_cpu_percent": cpu_percent,
                "system_memory_percent": vm.percent,
                "system_memory_available_gb": vm.available / 1024**3,
                "disk_usage_percent": disk_usage.percent,
                "disk_free_gb": disk_usage.free / 1024**3
            }

            return ComponentHealth(
                name="system_resources",
                status=status,
                message=message,
                details=details
            )

        except Exception as e:
            return ComponentHealth(
                name="system_resources",
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check system resources: {str(e)}",
                error=str(e)
            )

    def _check_python_runtime(self) -> ComponentHealth:
        """Check Python runtime status"""
        try:
            import sys
            import gc

            details = {
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "executable": sys.executable,
                "platform": sys.platform,
                "gc_counts": gc.get_count(),
                "recursion_limit": sys.getrecursionlimit(),
                "thread_count": threading.active_count() if 'threading' in globals() else "unknown"
            }

            # Check for issues
            if sys.version_info < (3, 10):
                return ComponentHealth(
                    name="python_runtime",
                    status=HealthStatus.DEGRADED,
                    message=f"Python version {details['python_version']} is below recommended 3.10+",
                    details=details
                )

            return ComponentHealth(
                name="python_runtime",
                status=HealthStatus.HEALTHY,
                message=f"Python runtime {details['python_version']} is healthy",
                details=details
            )

        except Exception as e:
            return ComponentHealth(
                name="python_runtime",
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check Python runtime: {str(e)}",
                error=str(e)
            )

    def _check_greenlang_core(self) -> ComponentHealth:
        """Check GreenLang core components"""
        try:
            # Check imports
            from ..runtime.executor import Executor
            from ..sdk.base import Result, Agent, Pipeline
            from ..packs.loader import PackLoader

            # Test basic functionality
            executor = Executor(backend="local")
            loader = PackLoader()

            details = {
                "executor_backend": executor.backend,
                "deterministic_support": executor.deterministic,
                "loader_initialized": True,
                "core_imports": "success"
            }

            return ComponentHealth(
                name="greenlang_core",
                status=HealthStatus.HEALTHY,
                message="GreenLang core components are operational",
                details=details
            )

        except ImportError as e:
            return ComponentHealth(
                name="greenlang_core",
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to import GreenLang core: {str(e)}",
                error=str(e)
            )
        except Exception as e:
            return ComponentHealth(
                name="greenlang_core",
                status=HealthStatus.DEGRADED,
                message=f"GreenLang core issue: {str(e)}",
                error=str(e)
            )

    def _check_metrics_collector(self) -> ComponentHealth:
        """Check metrics collection system"""
        try:
            metrics = get_metrics()

            if metrics is None:
                return ComponentHealth(
                    name="metrics_collector",
                    status=HealthStatus.DEGRADED,
                    message="Metrics collector not initialized"
                )

            # Get health metrics
            health_metrics = metrics.get_health_metrics()

            # Check for issues
            issues = []
            if health_metrics["errors_last_minute"] > 10:
                issues.append(f"High error rate: {health_metrics['errors_last_minute']} errors/min")

            if health_metrics["memory_usage_mb"] > 1000:  # 1GB threshold
                issues.append(f"High memory usage: {health_metrics['memory_usage_mb']:.1f}MB")

            if health_metrics["buffer_size"] > 9000:  # Near buffer limit
                issues.append("Metrics buffer near capacity")

            # Determine status
            if len(issues) >= 2:
                status = HealthStatus.UNHEALTHY
                message = f"Multiple metrics issues: {'; '.join(issues)}"
            elif issues:
                status = HealthStatus.DEGRADED
                message = issues[0]
            else:
                status = HealthStatus.HEALTHY
                message = "Metrics collection is healthy"

            return ComponentHealth(
                name="metrics_collector",
                status=status,
                message=message,
                details=health_metrics
            )

        except Exception as e:
            return ComponentHealth(
                name="metrics_collector",
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check metrics collector: {str(e)}",
                error=str(e)
            )


class SimpleHealthHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler for health checks"""

    def __init__(self, health_checker: HealthChecker, *args, **kwargs):
        self.health_checker = health_checker
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Handle GET requests"""
        path = urlparse(self.path).path

        if path == "/health":
            self._handle_health_check()
        elif path == "/health/live":
            self._handle_liveness_check()
        elif path == "/health/ready":
            self._handle_readiness_check()
        elif path.startswith("/health/component/"):
            component_name = path.replace("/health/component/", "")
            self._handle_component_check(component_name)
        else:
            self._send_response(404, {"error": "Not found"})

    def _handle_health_check(self):
        """Handle full health check"""
        try:
            # Run async health check in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            health = loop.run_until_complete(self.health_checker.get_system_health())
            loop.close()

            status_code = 200 if health.status == HealthStatus.HEALTHY else 503
            self._send_response(status_code, health.to_dict())

        except Exception as e:
            self._send_response(500, {"error": str(e)})

    def _handle_liveness_check(self):
        """Handle liveness probe (simple check)"""
        self._send_response(200, {
            "status": "alive",
            "timestamp": time.time(),
            "uptime_seconds": time.time() - self.health_checker.start_time
        })

    def _handle_readiness_check(self):
        """Handle readiness probe (component checks)"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            health = loop.run_until_complete(self.health_checker.get_system_health())
            loop.close()

            # Readiness is based on critical components only
            ready = health.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
            status_code = 200 if ready else 503

            self._send_response(status_code, {
                "ready": ready,
                "status": health.status.value,
                "components": len(health.components),
                "healthy_components": health.summary["healthy_components"]
            })

        except Exception as e:
            self._send_response(500, {"error": str(e), "ready": False})

    def _handle_component_check(self, component_name: str):
        """Handle individual component check"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            component_health = loop.run_until_complete(
                self.health_checker.check_component(component_name)
            )
            loop.close()

            status_code = 200 if component_health.status == HealthStatus.HEALTHY else 503
            self._send_response(status_code, component_health.to_dict())

        except Exception as e:
            self._send_response(500, {"error": str(e)})

    def _send_response(self, status_code: int, data: Dict[str, Any]):
        """Send JSON response"""
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())

    def log_message(self, format, *args):
        """Override to reduce logging noise"""
        pass


def create_simple_health_server(health_checker: HealthChecker, port: int = 8080) -> HTTPServer:
    """Create a simple HTTP server for health checks"""
    if not HAS_HTTP_SERVER:
        raise ImportError("HTTP server support not available")

    handler = lambda *args, **kwargs: SimpleHealthHandler(health_checker, *args, **kwargs)
    server = HTTPServer(('', port), handler)

    logger.info(f"Health check server created on port {port}")
    return server


def create_flask_health_app(health_checker: HealthChecker) -> Flask:
    """Create Flask app for health checks"""
    if not HAS_FLASK:
        raise ImportError("Flask is required for Flask health app")

    app = Flask(__name__)

    @app.route('/health')
    async def health():
        try:
            health_status = await health_checker.get_system_health()
            status_code = 200 if health_status.status == HealthStatus.HEALTHY else 503
            return jsonify(health_status.to_dict()), status_code
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/health/live')
    def liveness():
        return jsonify({
            "status": "alive",
            "timestamp": time.time(),
            "uptime_seconds": time.time() - health_checker.start_time
        })

    @app.route('/health/ready')
    async def readiness():
        try:
            health_status = await health_checker.get_system_health()
            ready = health_status.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
            status_code = 200 if ready else 503

            return jsonify({
                "ready": ready,
                "status": health_status.status.value,
                "components": len(health_status.components),
                "healthy_components": health_status.summary["healthy_components"]
            }), status_code
        except Exception as e:
            return jsonify({"error": str(e), "ready": False}), 500

    @app.route('/health/component/<component_name>')
    async def component_health(component_name: str):
        try:
            component_health = await health_checker.check_component(component_name)
            status_code = 200 if component_health.status == HealthStatus.HEALTHY else 503
            return jsonify(component_health.to_dict()), status_code
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


def create_fastapi_health_app(health_checker: HealthChecker) -> FastAPI:
    """Create FastAPI app for health checks"""
    if not HAS_FASTAPI:
        raise ImportError("FastAPI is required for FastAPI health app")

    app = FastAPI(title="GreenLang Health Checks", version="0.2.0")

    @app.get("/health")
    async def health():
        try:
            health_status = await health_checker.get_system_health()
            status_code = 200 if health_status.status == HealthStatus.HEALTHY else 503
            return Response(
                content=json.dumps(health_status.to_dict(), indent=2),
                status_code=status_code,
                media_type="application/json"
            )
        except Exception as e:
            return Response(
                content=json.dumps({"error": str(e)}),
                status_code=500,
                media_type="application/json"
            )

    @app.get("/health/live")
    async def liveness():
        return {
            "status": "alive",
            "timestamp": time.time(),
            "uptime_seconds": time.time() - health_checker.start_time
        }

    @app.get("/health/ready")
    async def readiness():
        try:
            health_status = await health_checker.get_system_health()
            ready = health_status.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
            status_code = 200 if ready else 503

            return Response(
                content=json.dumps({
                    "ready": ready,
                    "status": health_status.status.value,
                    "components": len(health_status.components),
                    "healthy_components": health_status.summary["healthy_components"]
                }),
                status_code=status_code,
                media_type="application/json"
            )
        except Exception as e:
            return Response(
                content=json.dumps({"error": str(e), "ready": False}),
                status_code=500,
                media_type="application/json"
            )

    @app.get("/health/component/{component_name}")
    async def component_health(component_name: str):
        try:
            component_health_result = await health_checker.check_component(component_name)
            status_code = 200 if component_health_result.status == HealthStatus.HEALTHY else 503
            return Response(
                content=json.dumps(component_health_result.to_dict(), indent=2),
                status_code=status_code,
                media_type="application/json"
            )
        except Exception as e:
            return Response(
                content=json.dumps({"error": str(e)}),
                status_code=500,
                media_type="application/json"
            )

    return app


def create_health_app(health_checker: HealthChecker, framework: str = "auto", port: int = 8080):
    """Create health check app with automatic framework detection"""
    if framework == "auto":
        if HAS_FASTAPI:
            framework = "fastapi"
        elif HAS_FLASK:
            framework = "flask"
        elif HAS_HTTP_SERVER:
            framework = "simple"
        else:
            raise ImportError("No supported web framework available")

    if framework == "fastapi":
        app = create_fastapi_health_app(health_checker)
        return lambda: uvicorn.run(app, host="0.0.0.0", port=port)

    elif framework == "flask":
        app = create_flask_health_app(health_checker)
        return lambda: app.run(host="0.0.0.0", port=port)

    elif framework == "simple":
        server = create_simple_health_server(health_checker, port)
        return lambda: server.serve_forever()

    else:
        raise ValueError(f"Unsupported framework: {framework}")


# Example usage and testing
if __name__ == "__main__":
    import threading

    async def test_health_checks():
        """Test health check system"""
        print("Setting up health checker...")

        health = HealthChecker(check_interval=10)

        # Add custom component check
        def check_custom_component():
            return HealthStatus.HEALTHY, "Custom component is working", {"test": True}

        health.add_component_check("custom_component", check_custom_component)

        print("Running health checks...")

        # Check individual components
        for component in health.component_checks:
            result = await health.check_component(component)
            print(f"  {component}: {result.status.value} - {result.message}")

        # Get overall system health
        system_health = await health.get_system_health()
        print(f"\nOverall Status: {system_health.status.value}")
        print(f"Components: {system_health.summary['total_components']}")
        print(f"Healthy: {system_health.summary['healthy_components']}")
        print(f"Uptime: {system_health.uptime_seconds:.1f}s")

        # Start health check server
        print(f"\nStarting health check server on port 8080...")

        try:
            app_runner = create_health_app(health, port=8080)

            # Run server in background thread
            server_thread = threading.Thread(target=app_runner, daemon=True)
            server_thread.start()

            print("Health endpoints available:")
            print("  http://localhost:8080/health - Full health check")
            print("  http://localhost:8080/health/live - Liveness probe")
            print("  http://localhost:8080/health/ready - Readiness probe")
            print("  http://localhost:8080/health/component/<name> - Component check")

            # Keep running for a while
            await asyncio.sleep(5)
            print("\nHealth check test completed!")

        except Exception as e:
            print(f"Failed to start health server: {e}")

    asyncio.run(test_health_checks())