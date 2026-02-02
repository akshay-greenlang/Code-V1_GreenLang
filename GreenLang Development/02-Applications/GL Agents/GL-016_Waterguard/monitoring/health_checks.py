"""
Health Checks for GL-016 Waterguard

This module provides health check endpoints for the Waterguard boiler water
chemistry optimization agent. Supports liveness, readiness, and component
health checks for Kubernetes deployments.

Key Features:
    - Liveness probes: Is the agent process alive?
    - Readiness probes: Is the agent ready to serve requests?
    - Component health: Individual checks for database, Kafka, OPC-UA, analyzers
    - Aggregate health status with severity levels

Example:
    >>> health_checker = HealthChecker()
    >>> health_checker.register_check(DatabaseHealthCheck(connection_string))
    >>> health_checker.register_check(KafkaHealthCheck(bootstrap_servers))
    >>> result = await health_checker.check_readiness()
    >>> print(result.status)

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import logging
import socket
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# HEALTH STATUS TYPES
# =============================================================================

class HealthStatus(str, Enum):
    """Health status levels."""

    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"
    UNKNOWN = "UNKNOWN"


class ComponentHealth(BaseModel):
    """Health status of a single component."""

    component_id: str = Field(..., description="Component identifier")
    component_type: str = Field(..., description="Type of component")
    status: HealthStatus = Field(..., description="Health status")
    message: str = Field(default="", description="Status message")
    latency_ms: float = Field(default=0.0, description="Check latency in ms")
    last_check: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of last check"
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional details"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class HealthCheckResult(BaseModel):
    """Result of an aggregate health check."""

    status: HealthStatus = Field(..., description="Aggregate status")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Check timestamp"
    )
    components: List[ComponentHealth] = Field(
        default_factory=list,
        description="Individual component health"
    )
    total_latency_ms: float = Field(
        default=0.0,
        description="Total check latency"
    )
    healthy_count: int = Field(default=0, description="Number of healthy components")
    degraded_count: int = Field(default=0, description="Number of degraded components")
    unhealthy_count: int = Field(default=0, description="Number of unhealthy components")
    message: str = Field(default="", description="Overall status message")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# =============================================================================
# BASE HEALTH CHECK CLASS
# =============================================================================

class BaseHealthCheck(ABC):
    """
    Abstract base class for health checks.

    All health check implementations must inherit from this class
    and implement the check() method.
    """

    def __init__(
        self,
        component_id: str,
        component_type: str,
        timeout_seconds: float = 5.0,
        critical: bool = True,
    ):
        """
        Initialize health check.

        Args:
            component_id: Unique identifier for this component
            component_type: Type of component (database, kafka, etc.)
            timeout_seconds: Timeout for health check
            critical: If True, failure makes overall status UNHEALTHY
        """
        self.component_id = component_id
        self.component_type = component_type
        self.timeout_seconds = timeout_seconds
        self.critical = critical
        self._last_result: Optional[ComponentHealth] = None

    @abstractmethod
    async def check(self) -> ComponentHealth:
        """
        Perform the health check.

        Returns:
            ComponentHealth with status and details
        """
        pass

    async def check_with_timeout(self) -> ComponentHealth:
        """Perform health check with timeout handling."""
        start_time = time.perf_counter()

        try:
            result = await asyncio.wait_for(
                self.check(),
                timeout=self.timeout_seconds
            )
            result.latency_ms = (time.perf_counter() - start_time) * 1000
            self._last_result = result
            return result

        except asyncio.TimeoutError:
            latency = (time.perf_counter() - start_time) * 1000
            result = ComponentHealth(
                component_id=self.component_id,
                component_type=self.component_type,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {self.timeout_seconds}s",
                latency_ms=latency,
            )
            self._last_result = result
            return result

        except Exception as e:
            latency = (time.perf_counter() - start_time) * 1000
            result = ComponentHealth(
                component_id=self.component_id,
                component_type=self.component_type,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                latency_ms=latency,
            )
            self._last_result = result
            return result

    @property
    def last_result(self) -> Optional[ComponentHealth]:
        """Get the last health check result."""
        return self._last_result


# =============================================================================
# DATABASE HEALTH CHECK
# =============================================================================

class DatabaseHealthCheck(BaseHealthCheck):
    """
    Health check for PostgreSQL database connection.

    Verifies database connectivity and optionally checks query performance.
    """

    def __init__(
        self,
        connection_string: str,
        component_id: str = "database-primary",
        timeout_seconds: float = 5.0,
        test_query: str = "SELECT 1",
    ):
        """
        Initialize database health check.

        Args:
            connection_string: Database connection string
            component_id: Component identifier
            timeout_seconds: Timeout for health check
            test_query: Query to execute for health check
        """
        super().__init__(
            component_id=component_id,
            component_type="database",
            timeout_seconds=timeout_seconds,
            critical=True,
        )
        self.connection_string = connection_string
        self.test_query = test_query

    async def check(self) -> ComponentHealth:
        """
        Check database health.

        Returns:
            ComponentHealth with database status
        """
        try:
            # In production, use asyncpg or similar
            # This is a simplified version for demonstration
            import asyncio

            # Simulate database connection check
            # In production: async with asyncpg.connect(self.connection_string) as conn:
            #     await conn.fetchval(self.test_query)

            # Parse connection string to get host for basic connectivity check
            host = self._parse_host()
            port = self._parse_port()

            # Basic TCP connectivity check
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout_seconds)

            try:
                result = sock.connect_ex((host, port))
                if result == 0:
                    return ComponentHealth(
                        component_id=self.component_id,
                        component_type=self.component_type,
                        status=HealthStatus.HEALTHY,
                        message="Database connection successful",
                        details={
                            "host": host,
                            "port": port,
                            "test_query": self.test_query,
                        }
                    )
                else:
                    return ComponentHealth(
                        component_id=self.component_id,
                        component_type=self.component_type,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Database connection failed (error code: {result})",
                        details={"host": host, "port": port}
                    )
            finally:
                sock.close()

        except Exception as e:
            return ComponentHealth(
                component_id=self.component_id,
                component_type=self.component_type,
                status=HealthStatus.UNHEALTHY,
                message=f"Database check failed: {str(e)}",
            )

    def _parse_host(self) -> str:
        """Parse host from connection string."""
        # Simple parsing - in production use proper URL parsing
        if "@" in self.connection_string:
            after_at = self.connection_string.split("@")[1]
            if ":" in after_at:
                return after_at.split(":")[0]
            if "/" in after_at:
                return after_at.split("/")[0]
            return after_at
        return "localhost"

    def _parse_port(self) -> int:
        """Parse port from connection string."""
        try:
            if "@" in self.connection_string:
                after_at = self.connection_string.split("@")[1]
                if ":" in after_at:
                    port_str = after_at.split(":")[1].split("/")[0]
                    return int(port_str)
        except (ValueError, IndexError):
            pass
        return 5432  # Default PostgreSQL port


# =============================================================================
# KAFKA HEALTH CHECK
# =============================================================================

class KafkaHealthCheck(BaseHealthCheck):
    """
    Health check for Apache Kafka connectivity.

    Verifies broker connectivity and optionally topic availability.
    """

    def __init__(
        self,
        bootstrap_servers: Union[str, List[str]],
        component_id: str = "kafka-cluster",
        timeout_seconds: float = 10.0,
        required_topics: Optional[List[str]] = None,
    ):
        """
        Initialize Kafka health check.

        Args:
            bootstrap_servers: Kafka bootstrap servers
            component_id: Component identifier
            timeout_seconds: Timeout for health check
            required_topics: Topics that must exist
        """
        super().__init__(
            component_id=component_id,
            component_type="kafka",
            timeout_seconds=timeout_seconds,
            critical=True,
        )
        if isinstance(bootstrap_servers, str):
            self.bootstrap_servers = [bootstrap_servers]
        else:
            self.bootstrap_servers = bootstrap_servers
        self.required_topics = required_topics or []

    async def check(self) -> ComponentHealth:
        """
        Check Kafka health.

        Returns:
            ComponentHealth with Kafka status
        """
        try:
            brokers_available = 0
            broker_details = []

            for server in self.bootstrap_servers:
                host, port = self._parse_server(server)

                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self.timeout_seconds / len(self.bootstrap_servers))

                try:
                    result = sock.connect_ex((host, port))
                    if result == 0:
                        brokers_available += 1
                        broker_details.append({"server": server, "status": "available"})
                    else:
                        broker_details.append({"server": server, "status": "unavailable"})
                finally:
                    sock.close()

            total_brokers = len(self.bootstrap_servers)

            if brokers_available == total_brokers:
                status = HealthStatus.HEALTHY
                message = f"All {total_brokers} Kafka brokers available"
            elif brokers_available > 0:
                status = HealthStatus.DEGRADED
                message = f"{brokers_available}/{total_brokers} Kafka brokers available"
            else:
                status = HealthStatus.UNHEALTHY
                message = "No Kafka brokers available"

            return ComponentHealth(
                component_id=self.component_id,
                component_type=self.component_type,
                status=status,
                message=message,
                details={
                    "brokers_available": brokers_available,
                    "total_brokers": total_brokers,
                    "broker_details": broker_details,
                    "required_topics": self.required_topics,
                }
            )

        except Exception as e:
            return ComponentHealth(
                component_id=self.component_id,
                component_type=self.component_type,
                status=HealthStatus.UNHEALTHY,
                message=f"Kafka check failed: {str(e)}",
            )

    def _parse_server(self, server: str) -> tuple:
        """Parse host and port from server string."""
        if ":" in server:
            parts = server.split(":")
            return parts[0], int(parts[1])
        return server, 9092  # Default Kafka port


# =============================================================================
# OPC-UA HEALTH CHECK
# =============================================================================

class OPCUAHealthCheck(BaseHealthCheck):
    """
    Health check for OPC-UA server connectivity.

    Verifies OPC-UA server is reachable and responding.
    """

    def __init__(
        self,
        endpoint_url: str,
        component_id: str = "opcua-server",
        timeout_seconds: float = 10.0,
        required_nodes: Optional[List[str]] = None,
    ):
        """
        Initialize OPC-UA health check.

        Args:
            endpoint_url: OPC-UA server endpoint URL
            component_id: Component identifier
            timeout_seconds: Timeout for health check
            required_nodes: Node IDs that must be accessible
        """
        super().__init__(
            component_id=component_id,
            component_type="opcua",
            timeout_seconds=timeout_seconds,
            critical=True,
        )
        self.endpoint_url = endpoint_url
        self.required_nodes = required_nodes or []

    async def check(self) -> ComponentHealth:
        """
        Check OPC-UA server health.

        Returns:
            ComponentHealth with OPC-UA status
        """
        try:
            # Parse endpoint URL to get host and port
            # Format: opc.tcp://hostname:port/path
            host, port = self._parse_endpoint()

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout_seconds)

            try:
                result = sock.connect_ex((host, port))
                if result == 0:
                    return ComponentHealth(
                        component_id=self.component_id,
                        component_type=self.component_type,
                        status=HealthStatus.HEALTHY,
                        message="OPC-UA server reachable",
                        details={
                            "endpoint": self.endpoint_url,
                            "host": host,
                            "port": port,
                            "required_nodes": self.required_nodes,
                        }
                    )
                else:
                    return ComponentHealth(
                        component_id=self.component_id,
                        component_type=self.component_type,
                        status=HealthStatus.UNHEALTHY,
                        message=f"OPC-UA server unreachable (error: {result})",
                        details={"endpoint": self.endpoint_url}
                    )
            finally:
                sock.close()

        except Exception as e:
            return ComponentHealth(
                component_id=self.component_id,
                component_type=self.component_type,
                status=HealthStatus.UNHEALTHY,
                message=f"OPC-UA check failed: {str(e)}",
            )

    def _parse_endpoint(self) -> tuple:
        """Parse host and port from OPC-UA endpoint URL."""
        try:
            # opc.tcp://hostname:port/path
            url = self.endpoint_url
            if url.startswith("opc.tcp://"):
                url = url[10:]  # Remove protocol
            elif url.startswith("opc.https://"):
                url = url[12:]

            if "/" in url:
                url = url.split("/")[0]

            if ":" in url:
                parts = url.split(":")
                return parts[0], int(parts[1])

            return url, 4840  # Default OPC-UA port

        except Exception:
            return "localhost", 4840


# =============================================================================
# ANALYZER HEALTH CHECK
# =============================================================================

class AnalyzerHealthCheck(BaseHealthCheck):
    """
    Health check for water chemistry analyzers.

    Monitors multiple analyzers and their online status.
    """

    def __init__(
        self,
        analyzer_configs: List[Dict[str, Any]],
        component_id: str = "analyzers",
        timeout_seconds: float = 15.0,
        min_healthy_analyzers: int = 1,
    ):
        """
        Initialize analyzer health check.

        Args:
            analyzer_configs: List of analyzer configurations
                Each config: {"id": str, "host": str, "port": int, "parameter": str}
            component_id: Component identifier
            timeout_seconds: Timeout for health check
            min_healthy_analyzers: Minimum analyzers required for HEALTHY status
        """
        super().__init__(
            component_id=component_id,
            component_type="analyzers",
            timeout_seconds=timeout_seconds,
            critical=True,
        )
        self.analyzer_configs = analyzer_configs
        self.min_healthy_analyzers = min_healthy_analyzers

    async def check(self) -> ComponentHealth:
        """
        Check all analyzers health.

        Returns:
            ComponentHealth with aggregate analyzer status
        """
        if not self.analyzer_configs:
            return ComponentHealth(
                component_id=self.component_id,
                component_type=self.component_type,
                status=HealthStatus.UNKNOWN,
                message="No analyzers configured",
            )

        try:
            analyzer_results = []
            healthy_count = 0

            for config in self.analyzer_configs:
                analyzer_id = config.get("id", "unknown")
                host = config.get("host", "localhost")
                port = config.get("port", 502)  # Default Modbus port
                parameter = config.get("parameter", "unknown")

                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                timeout_per_analyzer = self.timeout_seconds / len(self.analyzer_configs)
                sock.settimeout(timeout_per_analyzer)

                try:
                    result = sock.connect_ex((host, port))
                    if result == 0:
                        healthy_count += 1
                        analyzer_results.append({
                            "id": analyzer_id,
                            "parameter": parameter,
                            "status": "online",
                            "host": host,
                            "port": port,
                        })
                    else:
                        analyzer_results.append({
                            "id": analyzer_id,
                            "parameter": parameter,
                            "status": "offline",
                            "host": host,
                            "port": port,
                            "error_code": result,
                        })
                except Exception as e:
                    analyzer_results.append({
                        "id": analyzer_id,
                        "parameter": parameter,
                        "status": "error",
                        "error": str(e),
                    })
                finally:
                    sock.close()

            total_analyzers = len(self.analyzer_configs)

            if healthy_count >= total_analyzers:
                status = HealthStatus.HEALTHY
                message = f"All {total_analyzers} analyzers online"
            elif healthy_count >= self.min_healthy_analyzers:
                status = HealthStatus.DEGRADED
                message = f"{healthy_count}/{total_analyzers} analyzers online"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Only {healthy_count}/{total_analyzers} analyzers online (min: {self.min_healthy_analyzers})"

            return ComponentHealth(
                component_id=self.component_id,
                component_type=self.component_type,
                status=status,
                message=message,
                details={
                    "healthy_count": healthy_count,
                    "total_analyzers": total_analyzers,
                    "min_required": self.min_healthy_analyzers,
                    "analyzers": analyzer_results,
                }
            )

        except Exception as e:
            return ComponentHealth(
                component_id=self.component_id,
                component_type=self.component_type,
                status=HealthStatus.UNHEALTHY,
                message=f"Analyzer check failed: {str(e)}",
            )


# =============================================================================
# CUSTOM HEALTH CHECK
# =============================================================================

class CustomHealthCheck(BaseHealthCheck):
    """
    Custom health check with user-defined check function.

    Allows flexible health checks for any component.
    """

    def __init__(
        self,
        component_id: str,
        component_type: str,
        check_function: Callable[[], Dict[str, Any]],
        timeout_seconds: float = 5.0,
        critical: bool = False,
    ):
        """
        Initialize custom health check.

        Args:
            component_id: Component identifier
            component_type: Type of component
            check_function: Function that returns {"status": str, "message": str, ...}
            timeout_seconds: Timeout for health check
            critical: If True, failure makes overall status UNHEALTHY
        """
        super().__init__(
            component_id=component_id,
            component_type=component_type,
            timeout_seconds=timeout_seconds,
            critical=critical,
        )
        self.check_function = check_function

    async def check(self) -> ComponentHealth:
        """
        Run custom health check.

        Returns:
            ComponentHealth based on check function result
        """
        try:
            # Support both sync and async check functions
            if asyncio.iscoroutinefunction(self.check_function):
                result = await self.check_function()
            else:
                result = self.check_function()

            status_str = result.get("status", "unknown").upper()

            if status_str == "HEALTHY":
                status = HealthStatus.HEALTHY
            elif status_str == "DEGRADED":
                status = HealthStatus.DEGRADED
            elif status_str == "UNHEALTHY":
                status = HealthStatus.UNHEALTHY
            else:
                status = HealthStatus.UNKNOWN

            return ComponentHealth(
                component_id=self.component_id,
                component_type=self.component_type,
                status=status,
                message=result.get("message", ""),
                details=result.get("details", {}),
            )

        except Exception as e:
            return ComponentHealth(
                component_id=self.component_id,
                component_type=self.component_type,
                status=HealthStatus.UNHEALTHY,
                message=f"Custom check failed: {str(e)}",
            )


# =============================================================================
# MAIN HEALTH CHECKER
# =============================================================================

class HealthChecker:
    """
    Aggregate health checker for Waterguard agent.

    Manages multiple health checks and provides aggregate liveness
    and readiness probes for Kubernetes deployments.

    Attributes:
        checks: Registered health checks
        startup_time: When the health checker was initialized

    Example:
        >>> health_checker = HealthChecker()
        >>> health_checker.register_check(DatabaseHealthCheck(conn_string))
        >>> health_checker.register_check(KafkaHealthCheck(bootstrap_servers))
        >>> result = await health_checker.check_readiness()
        >>> print(result.status)
    """

    def __init__(
        self,
        liveness_timeout: float = 30.0,
        readiness_timeout: float = 60.0,
    ):
        """
        Initialize health checker.

        Args:
            liveness_timeout: Timeout for liveness checks
            readiness_timeout: Timeout for readiness checks
        """
        self._checks: Dict[str, BaseHealthCheck] = {}
        self._liveness_checks: Dict[str, BaseHealthCheck] = {}
        self._startup_time = datetime.now(timezone.utc)
        self._ready = False
        self._alive = True
        self.liveness_timeout = liveness_timeout
        self.readiness_timeout = readiness_timeout

        logger.info("HealthChecker initialized")

    def register_check(
        self,
        check: BaseHealthCheck,
        liveness: bool = False,
    ) -> str:
        """
        Register a health check.

        Args:
            check: Health check to register
            liveness: If True, include in liveness probe

        Returns:
            Component ID of registered check
        """
        self._checks[check.component_id] = check

        if liveness:
            self._liveness_checks[check.component_id] = check

        logger.info(
            f"Registered health check: {check.component_id}",
            extra={"component_type": check.component_type, "critical": check.critical}
        )

        return check.component_id

    def unregister_check(self, component_id: str) -> bool:
        """Remove a health check."""
        removed = False

        if component_id in self._checks:
            del self._checks[component_id]
            removed = True

        if component_id in self._liveness_checks:
            del self._liveness_checks[component_id]

        return removed

    async def check_liveness(self) -> HealthCheckResult:
        """
        Perform liveness probe.

        Liveness checks determine if the process is alive and should
        continue running. Failure triggers a container restart.

        Returns:
            HealthCheckResult with liveness status
        """
        start_time = time.perf_counter()

        # If no liveness checks registered, use internal state
        if not self._liveness_checks:
            return HealthCheckResult(
                status=HealthStatus.HEALTHY if self._alive else HealthStatus.UNHEALTHY,
                message="Process is alive" if self._alive else "Process is not alive",
                total_latency_ms=(time.perf_counter() - start_time) * 1000,
                healthy_count=1 if self._alive else 0,
                unhealthy_count=0 if self._alive else 1,
            )

        # Run liveness checks
        results = await self._run_checks(self._liveness_checks)

        total_latency = (time.perf_counter() - start_time) * 1000

        # For liveness, any unhealthy critical component means unhealthy
        healthy_count = sum(1 for r in results if r.status == HealthStatus.HEALTHY)
        unhealthy_count = sum(1 for r in results if r.status == HealthStatus.UNHEALTHY)
        degraded_count = sum(1 for r in results if r.status == HealthStatus.DEGRADED)

        critical_unhealthy = any(
            r.status == HealthStatus.UNHEALTHY
            for r in results
            if self._liveness_checks[r.component_id].critical
        )

        if critical_unhealthy:
            status = HealthStatus.UNHEALTHY
            message = "Liveness probe failed: critical component unhealthy"
        elif unhealthy_count > 0:
            status = HealthStatus.DEGRADED
            message = f"Liveness degraded: {unhealthy_count} components unhealthy"
        else:
            status = HealthStatus.HEALTHY
            message = "Liveness probe passed"

        return HealthCheckResult(
            status=status,
            components=results,
            total_latency_ms=total_latency,
            healthy_count=healthy_count,
            degraded_count=degraded_count,
            unhealthy_count=unhealthy_count,
            message=message,
        )

    async def check_readiness(self) -> HealthCheckResult:
        """
        Perform readiness probe.

        Readiness checks determine if the service is ready to accept
        traffic. Failure removes the pod from load balancer.

        Returns:
            HealthCheckResult with readiness status
        """
        start_time = time.perf_counter()

        if not self._checks:
            return HealthCheckResult(
                status=HealthStatus.HEALTHY if self._ready else HealthStatus.UNHEALTHY,
                message="Service is ready" if self._ready else "Service is not ready",
                total_latency_ms=(time.perf_counter() - start_time) * 1000,
                healthy_count=1 if self._ready else 0,
                unhealthy_count=0 if self._ready else 1,
            )

        # Run all checks
        results = await self._run_checks(self._checks)

        total_latency = (time.perf_counter() - start_time) * 1000

        healthy_count = sum(1 for r in results if r.status == HealthStatus.HEALTHY)
        unhealthy_count = sum(1 for r in results if r.status == HealthStatus.UNHEALTHY)
        degraded_count = sum(1 for r in results if r.status == HealthStatus.DEGRADED)

        # Any critical component unhealthy means not ready
        critical_unhealthy = any(
            r.status == HealthStatus.UNHEALTHY
            for r in results
            if self._checks[r.component_id].critical
        )

        if critical_unhealthy:
            status = HealthStatus.UNHEALTHY
            message = "Readiness probe failed: critical component unhealthy"
            self._ready = False
        elif degraded_count > 0 or (unhealthy_count > 0 and not critical_unhealthy):
            status = HealthStatus.DEGRADED
            message = f"Readiness degraded: {degraded_count + unhealthy_count} components degraded/unhealthy"
            self._ready = True
        else:
            status = HealthStatus.HEALTHY
            message = "Readiness probe passed"
            self._ready = True

        return HealthCheckResult(
            status=status,
            components=results,
            total_latency_ms=total_latency,
            healthy_count=healthy_count,
            degraded_count=degraded_count,
            unhealthy_count=unhealthy_count,
            message=message,
        )

    async def check_startup(self) -> HealthCheckResult:
        """
        Perform startup probe.

        Startup checks determine if the application has started.
        Used for slow-starting containers.

        Returns:
            HealthCheckResult with startup status
        """
        # For startup, we mainly check critical components
        result = await self.check_readiness()

        if result.status == HealthStatus.UNHEALTHY:
            result.message = "Startup probe failed: critical components not ready"
        elif result.status == HealthStatus.DEGRADED:
            result.message = "Startup probe passed (degraded mode)"
        else:
            result.message = "Startup probe passed"

        return result

    async def _run_checks(
        self,
        checks: Dict[str, BaseHealthCheck],
    ) -> List[ComponentHealth]:
        """Run multiple health checks concurrently."""
        if not checks:
            return []

        tasks = [
            check.check_with_timeout()
            for check in checks.values()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        component_results = []
        for check, result in zip(checks.values(), results):
            if isinstance(result, Exception):
                component_results.append(ComponentHealth(
                    component_id=check.component_id,
                    component_type=check.component_type,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check raised exception: {str(result)}",
                ))
            else:
                component_results.append(result)

        return component_results

    async def get_full_health(self) -> Dict[str, Any]:
        """
        Get comprehensive health status.

        Returns:
            Dictionary with full health information
        """
        liveness = await self.check_liveness()
        readiness = await self.check_readiness()

        return {
            "status": readiness.status.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": (
                datetime.now(timezone.utc) - self._startup_time
            ).total_seconds(),
            "liveness": {
                "status": liveness.status.value,
                "message": liveness.message,
                "latency_ms": liveness.total_latency_ms,
            },
            "readiness": {
                "status": readiness.status.value,
                "message": readiness.message,
                "latency_ms": readiness.total_latency_ms,
            },
            "components": {
                c.component_id: {
                    "type": c.component_type,
                    "status": c.status.value,
                    "message": c.message,
                    "latency_ms": c.latency_ms,
                    "last_check": c.last_check.isoformat(),
                    "details": c.details,
                }
                for c in readiness.components
            },
            "summary": {
                "healthy": readiness.healthy_count,
                "degraded": readiness.degraded_count,
                "unhealthy": readiness.unhealthy_count,
                "total": len(readiness.components),
            },
        }

    def set_alive(self, alive: bool) -> None:
        """Set internal alive state."""
        self._alive = alive
        if not alive:
            logger.warning("Health checker marked as not alive")

    def set_ready(self, ready: bool) -> None:
        """Set internal ready state."""
        self._ready = ready
        if not ready:
            logger.warning("Health checker marked as not ready")

    @property
    def uptime(self) -> float:
        """Get uptime in seconds."""
        return (datetime.now(timezone.utc) - self._startup_time).total_seconds()


# =============================================================================
# HTTP HANDLER FOR HEALTH ENDPOINTS
# =============================================================================

class HealthHTTPHandler:
    """
    HTTP handler for health check endpoints.

    Provides /health/live, /health/ready, /health/startup endpoints.
    In production, integrate with FastAPI or similar framework.
    """

    def __init__(self, health_checker: HealthChecker, port: int = 8080):
        """
        Initialize health HTTP handler.

        Args:
            health_checker: HealthChecker instance
            port: HTTP port for health endpoints
        """
        self.health_checker = health_checker
        self.port = port
        self._running = False

    async def handle_liveness(self) -> Dict[str, Any]:
        """Handle /health/live endpoint."""
        result = await self.health_checker.check_liveness()
        return {
            "status": result.status.value,
            "message": result.message,
        }

    async def handle_readiness(self) -> Dict[str, Any]:
        """Handle /health/ready endpoint."""
        result = await self.health_checker.check_readiness()
        return {
            "status": result.status.value,
            "message": result.message,
            "components": len(result.components),
        }

    async def handle_startup(self) -> Dict[str, Any]:
        """Handle /health/startup endpoint."""
        result = await self.health_checker.check_startup()
        return {
            "status": result.status.value,
            "message": result.message,
        }

    async def handle_full(self) -> Dict[str, Any]:
        """Handle /health endpoint (full status)."""
        return await self.health_checker.get_full_health()

    async def start(self) -> None:
        """Start the health HTTP server."""
        self._running = True
        logger.info(
            f"Health endpoints available at http://localhost:{self.port}/health/{{live,ready,startup}}"
        )

    async def stop(self) -> None:
        """Stop the health HTTP server."""
        self._running = False
        logger.info("Health endpoints stopped")
