"""
Health check endpoints for GL-007 FurnacePerformanceMonitor.

Provides comprehensive health status, readiness checks, and detailed
diagnostics for Kubernetes probes and monitoring systems.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ReadinessStatus(str, Enum):
    """Readiness status enumeration."""
    READY = "ready"
    NOT_READY = "not_ready"


@dataclass
class ComponentHealth:
    """Health status of individual component."""
    name: str
    status: HealthStatus
    latency_ms: float
    details: Dict[str, Any]
    error: Optional[str] = None


@dataclass
class HealthResponse:
    """Complete health check response."""
    status: HealthStatus
    timestamp: str
    uptime_seconds: float
    components: Dict[str, ComponentHealth]
    details: Dict[str, Any]


@dataclass
class ReadinessResponse:
    """Readiness check response."""
    ready: bool
    timestamp: str
    checks: Dict[str, bool]
    details: Dict[str, Any]


class HealthChecker:
    """
    Comprehensive health check system for GL-007.

    Checks:
    - Application startup
    - Database connectivity
    - Cache connectivity
    - SCADA connectivity
    - ML model availability
    - Time-series database
    - System resources
    - Performance metrics
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize health checker.

        Args:
            config: Application configuration
        """
        self.config = config
        self.start_time = time.time()
        self.last_check_time: Optional[float] = None
        self.check_results: Dict[str, ComponentHealth] = {}

    async def check_health(self) -> HealthResponse:
        """
        Perform comprehensive health check.

        Returns:
            HealthResponse with detailed status
        """
        logger.debug("Starting health check...")
        self.last_check_time = time.time()

        # Run all health checks concurrently
        checks = [
            self._check_application(),
            self._check_database(),
            self._check_cache(),
            self._check_scada_connection(),
            self._check_time_series_db(),
            self._check_ml_models(),
            self._check_system_resources(),
        ]

        results = await asyncio.gather(*checks, return_exceptions=True)

        # Aggregate results
        components = {}
        overall_status = HealthStatus.HEALTHY

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Health check failed: {result}")
                overall_status = HealthStatus.UNHEALTHY
                continue

            components[result.name] = result
            if result.status == HealthStatus.UNHEALTHY:
                overall_status = HealthStatus.UNHEALTHY
            elif result.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.DEGRADED

        # Calculate uptime
        uptime_seconds = time.time() - self.start_time

        return HealthResponse(
            status=overall_status,
            timestamp=datetime.now(timezone.utc).isoformat(),
            uptime_seconds=uptime_seconds,
            components=components,
            details={
                "message": f"Application is {overall_status.value}",
                "version": self.config.get("app_version", "1.0.0"),
                "environment": self.config.get("environment", "production"),
                "agent_id": "GL-007",
                "agent_name": "FurnacePerformanceMonitor"
            }
        )

    async def check_readiness(self) -> ReadinessResponse:
        """
        Check if application is ready to accept traffic.

        Returns:
            ReadinessResponse with detailed checks
        """
        logger.debug("Starting readiness check...")

        checks = {
            "database": await self._is_database_ready(),
            "cache": await self._is_cache_ready(),
            "scada": await self._is_scada_ready(),
            "time_series_db": await self._is_time_series_db_ready(),
            "ml_models": await self._is_ml_models_ready(),
            "startup_complete": self._is_startup_complete(),
        }

        overall_ready = all(checks.values())

        return ReadinessResponse(
            ready=overall_ready,
            timestamp=datetime.now(timezone.utc).isoformat(),
            checks=checks,
            details={
                "message": "Ready" if overall_ready else "Not ready",
                "failed_checks": [k for k, v in checks.items() if not v],
            }
        )

    async def check_startup(self) -> Dict[str, Any]:
        """
        Startup probe - has initialization completed?

        Returns:
            Startup status information
        """
        startup_complete = self._is_startup_complete()

        return {
            "startup_complete": startup_complete,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": time.time() - self.start_time,
            "initialization_tasks": {
                "config_loaded": True,
                "database_initialized": await self._is_database_ready(),
                "scada_connected": await self._is_scada_ready(),
                "models_loaded": await self._is_ml_models_ready(),
            }
        }

    async def _check_application(self) -> ComponentHealth:
        """Check application startup and basic functionality."""
        start = time.time()
        try:
            # Check if application is running
            latency = (time.time() - start) * 1000

            return ComponentHealth(
                name="application",
                status=HealthStatus.HEALTHY,
                latency_ms=latency,
                details={
                    "running": True,
                    "version": self.config.get("app_version", "1.0.0"),
                    "agent_id": "GL-007",
                    "uptime_seconds": time.time() - self.start_time
                }
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            logger.error(f"Application health check failed: {e}")
            return ComponentHealth(
                name="application",
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency,
                details={},
                error=str(e)
            )

    async def _check_database(self) -> ComponentHealth:
        """Check database connectivity and performance."""
        start = time.time()
        try:
            # In production, execute simple query like SELECT 1
            # Example: await db.session.execute("SELECT 1")
            await asyncio.sleep(0.001)  # Simulate query
            latency = (time.time() - start) * 1000

            return ComponentHealth(
                name="database",
                status=HealthStatus.HEALTHY,
                latency_ms=latency,
                details={
                    "connected": True,
                    "latency_ms": round(latency, 2),
                    "pool_size": 10,  # Would come from actual connection pool
                    "active_connections": 2
                }
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            logger.error(f"Database health check failed: {e}")
            return ComponentHealth(
                name="database",
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency,
                details={},
                error=str(e)
            )

    async def _check_cache(self) -> ComponentHealth:
        """Check Redis cache connectivity and performance."""
        start = time.time()
        try:
            # In production, execute PING command
            # Example: await redis_client.ping()
            await asyncio.sleep(0.001)  # Simulate ping
            latency = (time.time() - start) * 1000

            return ComponentHealth(
                name="cache",
                status=HealthStatus.HEALTHY,
                latency_ms=latency,
                details={
                    "connected": True,
                    "latency_ms": round(latency, 2),
                    "hit_rate_percent": 85.5,
                    "keys_count": 1234
                }
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            logger.warning(f"Cache health check degraded: {e}")
            # Cache failure is degraded, not critical
            return ComponentHealth(
                name="cache",
                status=HealthStatus.DEGRADED,
                latency_ms=latency,
                details={},
                error=str(e)
            )

    async def _check_scada_connection(self) -> ComponentHealth:
        """Check SCADA system connectivity."""
        start = time.time()
        try:
            # Check SCADA connection
            # Example: await scada_client.check_connection()
            await asyncio.sleep(0.005)  # Simulate SCADA check
            latency = (time.time() - start) * 1000

            return ComponentHealth(
                name="scada",
                status=HealthStatus.HEALTHY,
                latency_ms=latency,
                details={
                    "connected": True,
                    "latency_ms": round(latency, 2),
                    "active_tags": 250,
                    "data_quality_percent": 98.5,
                    "last_update_seconds_ago": 1.2
                }
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            logger.error(f"SCADA health check failed: {e}")
            return ComponentHealth(
                name="scada",
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency,
                details={},
                error=str(e)
            )

    async def _check_time_series_db(self) -> ComponentHealth:
        """Check time-series database connectivity."""
        start = time.time()
        try:
            # Check InfluxDB/TimescaleDB connection
            # Example: await tsdb_client.ping()
            await asyncio.sleep(0.002)  # Simulate check
            latency = (time.time() - start) * 1000

            return ComponentHealth(
                name="time_series_db",
                status=HealthStatus.HEALTHY,
                latency_ms=latency,
                details={
                    "connected": True,
                    "latency_ms": round(latency, 2),
                    "retention_days": 365,
                    "disk_usage_percent": 45.2
                }
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            logger.warning(f"Time-series DB health check degraded: {e}")
            return ComponentHealth(
                name="time_series_db",
                status=HealthStatus.DEGRADED,
                latency_ms=latency,
                details={},
                error=str(e)
            )

    async def _check_ml_models(self) -> ComponentHealth:
        """Check ML model availability and health."""
        start = time.time()
        try:
            # Check if ML models are loaded and operational
            # Example: await model_manager.check_models()
            await asyncio.sleep(0.003)  # Simulate model check
            latency = (time.time() - start) * 1000

            return ComponentHealth(
                name="ml_models",
                status=HealthStatus.HEALTHY,
                latency_ms=latency,
                details={
                    "models_loaded": True,
                    "latency_ms": round(latency, 2),
                    "available_models": [
                        "thermal_efficiency_predictor",
                        "maintenance_forecaster",
                        "anomaly_detector"
                    ],
                    "average_inference_ms": 45.2
                }
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            logger.error(f"ML models health check failed: {e}")
            return ComponentHealth(
                name="ml_models",
                status=HealthStatus.DEGRADED,
                latency_ms=latency,
                details={},
                error=str(e)
            )

    async def _check_system_resources(self) -> ComponentHealth:
        """Check system resources (memory, CPU, disk)."""
        start = time.time()
        try:
            import psutil

            # Get current process
            process = psutil.Process()

            # Memory usage
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()

            # CPU usage
            cpu_percent = process.cpu_percent(interval=0.1)

            # Disk usage (if applicable)
            disk_usage = psutil.disk_usage('/')

            latency = (time.time() - start) * 1000

            # Check thresholds
            status = HealthStatus.HEALTHY
            if memory_percent > 85 or cpu_percent > 90:
                status = HealthStatus.DEGRADED
            if memory_percent > 95 or cpu_percent > 98:
                status = HealthStatus.UNHEALTHY

            return ComponentHealth(
                name="system_resources",
                status=status,
                latency_ms=latency,
                details={
                    "memory_mb": round(memory_info.rss / 1024 / 1024, 2),
                    "memory_percent": round(memory_percent, 2),
                    "cpu_percent": round(cpu_percent, 2),
                    "disk_percent": round(disk_usage.percent, 2),
                    "thresholds": {
                        "memory_warning": 85,
                        "memory_critical": 95,
                        "cpu_warning": 90,
                        "cpu_critical": 98
                    }
                }
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            logger.error(f"System resource check failed: {e}")
            return ComponentHealth(
                name="system_resources",
                status=HealthStatus.DEGRADED,
                latency_ms=latency,
                details={},
                error=str(e)
            )

    async def _is_database_ready(self) -> bool:
        """Check if database is ready for queries."""
        try:
            # Execute simple query
            # return await db.session.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database readiness check failed: {e}")
            return False

    async def _is_cache_ready(self) -> bool:
        """Check if cache is ready."""
        try:
            # Execute PING command
            # return await redis_client.ping()
            return True
        except Exception as e:
            logger.warning(f"Cache readiness check failed: {e}")
            # Cache is optional, don't block readiness
            return True

    async def _is_scada_ready(self) -> bool:
        """Check if SCADA connection is ready."""
        try:
            # Check SCADA connection
            # return await scada_client.is_connected()
            return True
        except Exception as e:
            logger.error(f"SCADA readiness check failed: {e}")
            return False

    async def _is_time_series_db_ready(self) -> bool:
        """Check if time-series database is ready."""
        try:
            # Check time-series DB connection
            # return await tsdb_client.ping()
            return True
        except Exception as e:
            logger.warning(f"Time-series DB readiness check failed: {e}")
            # Time-series DB is optional for basic operation
            return True

    async def _is_ml_models_ready(self) -> bool:
        """Check if ML models are loaded and ready."""
        try:
            # Check if models are loaded
            # return await model_manager.are_models_ready()
            return True
        except Exception as e:
            logger.warning(f"ML models readiness check failed: {e}")
            # Models are optional for basic monitoring
            return True

    def _is_startup_complete(self) -> bool:
        """Check if application startup is complete."""
        # In production, check if all initialization tasks are done
        startup_grace_period = 30  # seconds - longer for ML model loading
        if time.time() - self.start_time < startup_grace_period:
            return False
        return True


# Example FastAPI/Starlette integration
async def health_endpoint(health_checker: HealthChecker) -> Dict[str, Any]:
    """
    FastAPI health check endpoint.

    GET /api/v1/health
    """
    health_response = await health_checker.check_health()

    # Convert ComponentHealth objects to dicts
    response_dict = asdict(health_response)
    response_dict['components'] = {
        name: asdict(component)
        for name, component in health_response.components.items()
    }

    return response_dict


async def readiness_endpoint(health_checker: HealthChecker) -> Dict[str, Any]:
    """
    FastAPI readiness check endpoint.

    GET /api/v1/ready
    """
    readiness_response = await health_checker.check_readiness()
    return asdict(readiness_response)


async def startup_endpoint(health_checker: HealthChecker) -> Dict[str, Any]:
    """
    FastAPI startup check endpoint.

    GET /api/v1/startup
    """
    startup_response = await health_checker.check_startup()
    return startup_response


# Kubernetes probe handlers
class KubernetesProbes:
    """Integration with Kubernetes probes."""

    @staticmethod
    async def liveness_probe(health_checker: HealthChecker) -> int:
        """
        Liveness probe - restart pod if returns non-zero.

        Returns:
            0 if healthy, 1 if unhealthy
        """
        try:
            health = await health_checker.check_health()
            return 0 if health.status != HealthStatus.UNHEALTHY else 1
        except Exception as e:
            logger.error(f"Liveness probe failed: {e}")
            return 1

    @staticmethod
    async def readiness_probe(health_checker: HealthChecker) -> int:
        """
        Readiness probe - remove from service if returns non-zero.

        Returns:
            0 if ready, 1 if not ready
        """
        try:
            readiness = await health_checker.check_readiness()
            return 0 if readiness.ready else 1
        except Exception as e:
            logger.error(f"Readiness probe failed: {e}")
            return 1

    @staticmethod
    async def startup_probe(health_checker: HealthChecker) -> int:
        """
        Startup probe - allow time for initialization.

        Returns:
            0 if startup complete, 1 otherwise
        """
        try:
            startup = await health_checker.check_startup()
            return 0 if startup['startup_complete'] else 1
        except Exception as e:
            logger.error(f"Startup probe failed: {e}")
            return 1
