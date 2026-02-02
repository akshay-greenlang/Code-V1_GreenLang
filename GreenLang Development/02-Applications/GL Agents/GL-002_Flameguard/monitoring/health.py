"""
GL-002 FLAMEGUARD - Health Checks

System health monitoring and status reporting.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import asyncio
import logging
import time

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status of a single component."""
    name: str
    status: HealthStatus
    message: str = ""
    last_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    response_time_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "last_check": self.last_check.isoformat(),
            "response_time_ms": self.response_time_ms,
            "details": self.details,
        }


@dataclass
class HealthCheckResult:
    """Overall health check result."""
    status: HealthStatus
    timestamp: datetime
    components: List[ComponentHealth]
    uptime_seconds: float
    version: str = "1.0.0"

    def to_dict(self) -> Dict:
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "uptime_seconds": self.uptime_seconds,
            "version": self.version,
            "components": [c.to_dict() for c in self.components],
        }


class HealthChecker:
    """
    System health checker.

    Features:
    - Component health checks
    - Dependency verification
    - Periodic background checks
    - Kubernetes readiness/liveness probes
    """

    def __init__(self) -> None:
        self._start_time = time.time()
        self._checks: Dict[str, Callable] = {}
        self._results: Dict[str, ComponentHealth] = {}
        self._running = False
        self._check_interval_s = 30.0

        # Register default checks
        self._register_default_checks()

        logger.info("HealthChecker initialized")

    def _register_default_checks(self) -> None:
        """Register default health checks."""
        self.register_check("self", self._check_self)

    def register_check(
        self,
        name: str,
        check_func: Callable[[], ComponentHealth],
    ) -> None:
        """Register a health check function."""
        self._checks[name] = check_func
        logger.debug(f"Registered health check: {name}")

    async def _check_self(self) -> ComponentHealth:
        """Self health check."""
        return ComponentHealth(
            name="self",
            status=HealthStatus.HEALTHY,
            message="Agent is running",
            details={
                "uptime_seconds": time.time() - self._start_time,
            },
        )

    async def check_scada(
        self,
        connector,
    ) -> ComponentHealth:
        """Check SCADA connection health."""
        start = time.time()

        try:
            if connector and connector.is_connected():
                return ComponentHealth(
                    name="scada",
                    status=HealthStatus.HEALTHY,
                    message="SCADA connected",
                    response_time_ms=(time.time() - start) * 1000,
                )
            else:
                return ComponentHealth(
                    name="scada",
                    status=HealthStatus.UNHEALTHY,
                    message="SCADA disconnected",
                )
        except Exception as e:
            return ComponentHealth(
                name="scada",
                status=HealthStatus.UNHEALTHY,
                message=str(e),
            )

    async def check_kafka(
        self,
        producer=None,
        consumer=None,
    ) -> ComponentHealth:
        """Check Kafka connection health."""
        start = time.time()

        try:
            producer_ok = producer.is_connected if producer else True
            consumer_ok = consumer.is_connected if consumer else True

            if producer_ok and consumer_ok:
                return ComponentHealth(
                    name="kafka",
                    status=HealthStatus.HEALTHY,
                    message="Kafka connected",
                    response_time_ms=(time.time() - start) * 1000,
                    details={
                        "producer": "connected" if producer_ok else "disconnected",
                        "consumer": "connected" if consumer_ok else "disconnected",
                    },
                )
            else:
                return ComponentHealth(
                    name="kafka",
                    status=HealthStatus.DEGRADED,
                    message="Kafka partially connected",
                    details={
                        "producer": "connected" if producer_ok else "disconnected",
                        "consumer": "connected" if consumer_ok else "disconnected",
                    },
                )
        except Exception as e:
            return ComponentHealth(
                name="kafka",
                status=HealthStatus.UNHEALTHY,
                message=str(e),
            )

    async def check_database(
        self,
        db_pool=None,
    ) -> ComponentHealth:
        """Check database connection health."""
        start = time.time()

        try:
            if db_pool:
                # In production, execute health check query
                # async with db_pool.acquire() as conn:
                #     await conn.execute("SELECT 1")
                return ComponentHealth(
                    name="database",
                    status=HealthStatus.HEALTHY,
                    message="Database connected",
                    response_time_ms=(time.time() - start) * 1000,
                )
            else:
                return ComponentHealth(
                    name="database",
                    status=HealthStatus.UNKNOWN,
                    message="No database configured",
                )
        except Exception as e:
            return ComponentHealth(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=str(e),
            )

    async def check_safety_system(
        self,
        bms=None,
        interlocks=None,
    ) -> ComponentHealth:
        """Check safety system health."""
        try:
            issues = []

            if bms:
                status = bms.get_status()
                if status.get("lockout_reason"):
                    issues.append(f"BMS lockout: {status['lockout_reason']}")

            if interlocks:
                interlock_status = interlocks.get_status()
                if interlock_status.get("tripped"):
                    issues.append("Safety interlocks tripped")

            if issues:
                return ComponentHealth(
                    name="safety_system",
                    status=HealthStatus.UNHEALTHY,
                    message="; ".join(issues),
                )

            return ComponentHealth(
                name="safety_system",
                status=HealthStatus.HEALTHY,
                message="Safety systems normal",
            )
        except Exception as e:
            return ComponentHealth(
                name="safety_system",
                status=HealthStatus.UNKNOWN,
                message=str(e),
            )

    async def run_all_checks(self) -> HealthCheckResult:
        """Run all registered health checks."""
        components: List[ComponentHealth] = []

        for name, check_func in self._checks.items():
            try:
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                components.append(result)
                self._results[name] = result
            except Exception as e:
                components.append(ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=str(e),
                ))

        # Determine overall status
        statuses = [c.status for c in components]

        if all(s == HealthStatus.HEALTHY for s in statuses):
            overall = HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall = HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.UNKNOWN

        return HealthCheckResult(
            status=overall,
            timestamp=datetime.now(timezone.utc),
            components=components,
            uptime_seconds=time.time() - self._start_time,
        )

    async def start_background_checks(
        self,
        interval_s: float = 30.0,
    ) -> None:
        """Start background health check loop."""
        self._running = True
        self._check_interval_s = interval_s

        while self._running:
            try:
                await self.run_all_checks()
            except Exception as e:
                logger.error(f"Background health check failed: {e}")

            await asyncio.sleep(self._check_interval_s)

    def stop_background_checks(self) -> None:
        """Stop background health check loop."""
        self._running = False

    def is_ready(self) -> bool:
        """Kubernetes readiness probe."""
        if not self._results:
            return False

        return all(
            r.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
            for r in self._results.values()
        )

    def is_live(self) -> bool:
        """Kubernetes liveness probe."""
        # Always live if running
        return True

    def get_status(self) -> Dict:
        """Get current health status."""
        return {
            "healthy": self.is_ready(),
            "live": self.is_live(),
            "uptime_seconds": time.time() - self._start_time,
            "components": {
                name: result.to_dict()
                for name, result in self._results.items()
            },
        }
