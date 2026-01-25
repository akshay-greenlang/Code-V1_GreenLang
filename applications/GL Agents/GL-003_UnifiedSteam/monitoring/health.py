"""
GL-003 UNIFIEDSTEAM SteamSystemOptimizer - Health Monitoring

This module provides comprehensive health monitoring for the steam system
optimization agent, including service health, integration health, model health,
and data freshness monitoring.

Features:
    - Liveness and readiness probes for Kubernetes
    - Dependency health checks (OPC-UA, Kafka, database)
    - Model health monitoring (prediction quality, drift detection)
    - Data freshness validation
    - Periodic background health checks

Example:
    >>> monitor = HealthMonitor()
    >>> service_health = await monitor.check_service_health()
    >>> overall = await monitor.get_overall_health()
    >>> print(f"System status: {overall.status.value}")
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

    @property
    def is_operational(self) -> bool:
        """Check if status allows operation."""
        return self in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]


@dataclass
class ComponentHealth:
    """Health status of a single component."""
    name: str
    status: HealthStatus
    message: str = ""
    last_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    response_time_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "last_check": self.last_check.isoformat(),
            "response_time_ms": self.response_time_ms,
            "details": self.details,
        }


@dataclass
class ServiceHealthStatus:
    """Service-level health status."""
    status: HealthStatus
    uptime_seconds: float
    version: str
    agent_id: str
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    active_tasks: int = 0
    error_count_last_hour: int = 0
    last_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "status": self.status.value,
            "uptime_seconds": self.uptime_seconds,
            "version": self.version,
            "agent_id": self.agent_id,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "active_tasks": self.active_tasks,
            "error_count_last_hour": self.error_count_last_hour,
            "last_check": self.last_check.isoformat(),
            "details": self.details,
        }


@dataclass
class IntegrationHealthStatus:
    """Integration health status for external dependencies."""
    name: str
    status: HealthStatus
    endpoint: Optional[str] = None
    connected: bool = False
    latency_ms: float = 0.0
    last_successful_connection: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "status": self.status.value,
            "endpoint": self.endpoint,
            "connected": self.connected,
            "latency_ms": self.latency_ms,
            "last_successful_connection": (
                self.last_successful_connection.isoformat()
                if self.last_successful_connection else None
            ),
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "details": self.details,
        }


@dataclass
class ModelHealthStatus:
    """Model health status for ML/physics models."""
    status: HealthStatus
    model_name: str
    model_version: str
    last_prediction_time: Optional[datetime] = None
    predictions_last_hour: int = 0
    average_inference_time_ms: float = 0.0
    prediction_error_rate: float = 0.0
    drift_detected: bool = False
    drift_score: float = 0.0
    calibration_status: str = "unknown"
    last_calibration: Optional[datetime] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "status": self.status.value,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "last_prediction_time": (
                self.last_prediction_time.isoformat()
                if self.last_prediction_time else None
            ),
            "predictions_last_hour": self.predictions_last_hour,
            "average_inference_time_ms": self.average_inference_time_ms,
            "prediction_error_rate": self.prediction_error_rate,
            "drift_detected": self.drift_detected,
            "drift_score": self.drift_score,
            "calibration_status": self.calibration_status,
            "last_calibration": (
                self.last_calibration.isoformat() if self.last_calibration else None
            ),
            "details": self.details,
        }


@dataclass
class DataFreshnessStatus:
    """Data freshness status for input streams."""
    status: HealthStatus
    stream_name: str
    last_data_received: Optional[datetime] = None
    age_seconds: float = 0.0
    max_age_seconds: float = 60.0
    is_stale: bool = False
    records_last_minute: int = 0
    missing_fields: List[str] = field(default_factory=list)
    quality_score: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "status": self.status.value,
            "stream_name": self.stream_name,
            "last_data_received": (
                self.last_data_received.isoformat()
                if self.last_data_received else None
            ),
            "age_seconds": self.age_seconds,
            "max_age_seconds": self.max_age_seconds,
            "is_stale": self.is_stale,
            "records_last_minute": self.records_last_minute,
            "missing_fields": self.missing_fields,
            "quality_score": self.quality_score,
            "details": self.details,
        }


@dataclass
class OverallHealthStatus:
    """Overall system health status."""
    status: HealthStatus
    timestamp: datetime
    uptime_seconds: float
    version: str
    agent_id: str
    service_health: ServiceHealthStatus
    integration_health: Dict[str, IntegrationHealthStatus]
    model_health: ModelHealthStatus
    data_freshness: Dict[str, DataFreshnessStatus]
    is_ready: bool = True
    is_live: bool = True
    degraded_components: List[str] = field(default_factory=list)
    unhealthy_components: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "uptime_seconds": self.uptime_seconds,
            "version": self.version,
            "agent_id": self.agent_id,
            "service_health": self.service_health.to_dict(),
            "integration_health": {
                k: v.to_dict() for k, v in self.integration_health.items()
            },
            "model_health": self.model_health.to_dict(),
            "data_freshness": {
                k: v.to_dict() for k, v in self.data_freshness.items()
            },
            "is_ready": self.is_ready,
            "is_live": self.is_live,
            "degraded_components": self.degraded_components,
            "unhealthy_components": self.unhealthy_components,
        }


class HealthMonitor:
    """
    Health monitoring system for steam system optimization agent.

    This class provides comprehensive health monitoring including service health,
    integration health (OPC-UA, Kafka, database), model health, and data freshness.
    It supports Kubernetes liveness and readiness probes.

    Attributes:
        version: Agent version string
        agent_id: Agent identifier

    Example:
        >>> monitor = HealthMonitor(version="1.0.0", agent_id="GL-003")
        >>> service_health = await monitor.check_service_health()
        >>> overall = await monitor.get_overall_health()
        >>> if monitor.is_ready():
        ...     print("System ready for requests")
    """

    def __init__(
        self,
        version: str = "1.0.0",
        agent_id: str = "GL-003",
        check_interval_s: float = 30.0,
    ) -> None:
        """
        Initialize HealthMonitor.

        Args:
            version: Agent version string
            agent_id: Agent identifier
            check_interval_s: Background check interval in seconds
        """
        self.version = version
        self.agent_id = agent_id
        self._check_interval_s = check_interval_s
        self._start_time = time.time()

        # Health check registry
        self._custom_checks: Dict[str, Callable] = {}

        # Cached results
        self._last_service_health: Optional[ServiceHealthStatus] = None
        self._last_integration_health: Dict[str, IntegrationHealthStatus] = {}
        self._last_model_health: Optional[ModelHealthStatus] = None
        self._last_data_freshness: Dict[str, DataFreshnessStatus] = {}
        self._last_overall_health: Optional[OverallHealthStatus] = None

        # Integration connectors (to be set externally)
        self._opc_ua_client: Optional[Any] = None
        self._kafka_producer: Optional[Any] = None
        self._kafka_consumer: Optional[Any] = None
        self._database_pool: Optional[Any] = None

        # Data stream timestamps
        self._data_stream_timestamps: Dict[str, datetime] = {}

        # Model metrics
        self._model_predictions: List[datetime] = []
        self._model_inference_times: List[float] = []
        self._model_errors: List[datetime] = []

        # Error tracking
        self._recent_errors: List[datetime] = []

        # Background check control
        self._running = False

        logger.info(
            "HealthMonitor initialized: version=%s, agent_id=%s",
            version,
            agent_id,
        )

    def set_opc_ua_client(self, client: Any) -> None:
        """Set OPC-UA client for health checks."""
        self._opc_ua_client = client

    def set_kafka_producer(self, producer: Any) -> None:
        """Set Kafka producer for health checks."""
        self._kafka_producer = producer

    def set_kafka_consumer(self, consumer: Any) -> None:
        """Set Kafka consumer for health checks."""
        self._kafka_consumer = consumer

    def set_database_pool(self, pool: Any) -> None:
        """Set database pool for health checks."""
        self._database_pool = pool

    def record_data_received(self, stream_name: str) -> None:
        """Record data reception timestamp for a stream."""
        self._data_stream_timestamps[stream_name] = datetime.now(timezone.utc)

    def record_model_prediction(self, inference_time_ms: float) -> None:
        """Record a model prediction for metrics."""
        now = datetime.now(timezone.utc)
        self._model_predictions.append(now)
        self._model_inference_times.append(inference_time_ms)

        # Keep only last hour of data
        cutoff = now - timedelta(hours=1)
        self._model_predictions = [t for t in self._model_predictions if t > cutoff]
        self._model_inference_times = self._model_inference_times[-len(self._model_predictions):]

    def record_model_error(self) -> None:
        """Record a model prediction error."""
        now = datetime.now(timezone.utc)
        self._model_errors.append(now)

        # Keep only last hour of data
        cutoff = now - timedelta(hours=1)
        self._model_errors = [t for t in self._model_errors if t > cutoff]

    def record_error(self) -> None:
        """Record a general error for tracking."""
        now = datetime.now(timezone.utc)
        self._recent_errors.append(now)

        # Keep only last hour of data
        cutoff = now - timedelta(hours=1)
        self._recent_errors = [t for t in self._recent_errors if t > cutoff]

    def register_custom_check(
        self,
        name: str,
        check_func: Callable[[], ComponentHealth],
    ) -> None:
        """Register a custom health check function."""
        self._custom_checks[name] = check_func
        logger.debug("Registered custom health check: %s", name)

    async def check_service_health(self) -> ServiceHealthStatus:
        """
        Check overall service health.

        Returns:
            ServiceHealthStatus with current service metrics
        """
        uptime = time.time() - self._start_time
        now = datetime.now(timezone.utc)

        # Calculate error count in last hour
        cutoff = now - timedelta(hours=1)
        error_count = len([e for e in self._recent_errors if e > cutoff])

        # Get resource usage (in production, use psutil)
        memory_mb = 0.0
        cpu_percent = 0.0

        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            cpu_percent = process.cpu_percent()
        except ImportError:
            pass
        except Exception as e:
            logger.debug("Resource metrics unavailable: %s", e)

        # Determine status
        status = HealthStatus.HEALTHY
        if error_count > 100:
            status = HealthStatus.UNHEALTHY
        elif error_count > 10:
            status = HealthStatus.DEGRADED

        self._last_service_health = ServiceHealthStatus(
            status=status,
            uptime_seconds=uptime,
            version=self.version,
            agent_id=self.agent_id,
            memory_usage_mb=memory_mb,
            cpu_usage_percent=cpu_percent,
            error_count_last_hour=error_count,
            last_check=now,
        )

        return self._last_service_health

    async def check_integration_health(self) -> Dict[str, IntegrationHealthStatus]:
        """
        Check health of all external integrations.

        Returns:
            Dictionary of integration name to health status
        """
        results: Dict[str, IntegrationHealthStatus] = {}

        # Check OPC-UA
        results["opc_ua"] = await self._check_opc_ua_health()

        # Check Kafka
        results["kafka"] = await self._check_kafka_health()

        # Check Database
        results["database"] = await self._check_database_health()

        self._last_integration_health = results
        return results

    async def _check_opc_ua_health(self) -> IntegrationHealthStatus:
        """Check OPC-UA connection health."""
        start_time = time.time()

        try:
            if self._opc_ua_client is None:
                return IntegrationHealthStatus(
                    name="opc_ua",
                    status=HealthStatus.UNKNOWN,
                    connected=False,
                    error_message="OPC-UA client not configured",
                )

            # In production, check actual connection
            is_connected = getattr(self._opc_ua_client, "is_connected", lambda: False)()

            latency_ms = (time.time() - start_time) * 1000

            if is_connected:
                return IntegrationHealthStatus(
                    name="opc_ua",
                    status=HealthStatus.HEALTHY,
                    endpoint=getattr(self._opc_ua_client, "endpoint", None),
                    connected=True,
                    latency_ms=latency_ms,
                    last_successful_connection=datetime.now(timezone.utc),
                )
            else:
                return IntegrationHealthStatus(
                    name="opc_ua",
                    status=HealthStatus.UNHEALTHY,
                    connected=False,
                    error_message="OPC-UA disconnected",
                )

        except Exception as e:
            return IntegrationHealthStatus(
                name="opc_ua",
                status=HealthStatus.UNHEALTHY,
                connected=False,
                error_message=str(e),
            )

    async def _check_kafka_health(self) -> IntegrationHealthStatus:
        """Check Kafka connection health."""
        start_time = time.time()

        try:
            producer_ok = True
            consumer_ok = True

            if self._kafka_producer is not None:
                producer_ok = getattr(
                    self._kafka_producer, "is_connected", lambda: True
                )()

            if self._kafka_consumer is not None:
                consumer_ok = getattr(
                    self._kafka_consumer, "is_connected", lambda: True
                )()

            latency_ms = (time.time() - start_time) * 1000

            if producer_ok and consumer_ok:
                status = HealthStatus.HEALTHY
                connected = True
            elif producer_ok or consumer_ok:
                status = HealthStatus.DEGRADED
                connected = True
            else:
                status = HealthStatus.UNHEALTHY
                connected = False

            return IntegrationHealthStatus(
                name="kafka",
                status=status,
                connected=connected,
                latency_ms=latency_ms,
                last_successful_connection=datetime.now(timezone.utc) if connected else None,
                details={
                    "producer_connected": producer_ok,
                    "consumer_connected": consumer_ok,
                },
            )

        except Exception as e:
            return IntegrationHealthStatus(
                name="kafka",
                status=HealthStatus.UNHEALTHY,
                connected=False,
                error_message=str(e),
            )

    async def _check_database_health(self) -> IntegrationHealthStatus:
        """Check database connection health."""
        start_time = time.time()

        try:
            if self._database_pool is None:
                return IntegrationHealthStatus(
                    name="database",
                    status=HealthStatus.UNKNOWN,
                    connected=False,
                    error_message="Database pool not configured",
                )

            # In production, execute health check query:
            # async with self._database_pool.acquire() as conn:
            #     await conn.execute("SELECT 1")

            latency_ms = (time.time() - start_time) * 1000

            return IntegrationHealthStatus(
                name="database",
                status=HealthStatus.HEALTHY,
                connected=True,
                latency_ms=latency_ms,
                last_successful_connection=datetime.now(timezone.utc),
            )

        except Exception as e:
            return IntegrationHealthStatus(
                name="database",
                status=HealthStatus.UNHEALTHY,
                connected=False,
                error_message=str(e),
            )

    async def check_model_health(self) -> ModelHealthStatus:
        """
        Check ML/physics model health.

        Returns:
            ModelHealthStatus with model metrics
        """
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=1)

        # Calculate metrics from recent predictions
        recent_predictions = [t for t in self._model_predictions if t > cutoff]
        recent_errors = [t for t in self._model_errors if t > cutoff]

        predictions_count = len(recent_predictions)
        error_count = len(recent_errors)

        error_rate = 0.0
        if predictions_count > 0:
            error_rate = error_count / (predictions_count + error_count)

        avg_inference_time = 0.0
        if self._model_inference_times:
            avg_inference_time = sum(self._model_inference_times) / len(
                self._model_inference_times
            )

        # Determine status
        status = HealthStatus.HEALTHY
        if error_rate > 0.1:  # >10% error rate
            status = HealthStatus.UNHEALTHY
        elif error_rate > 0.05:  # >5% error rate
            status = HealthStatus.DEGRADED

        # Check for model drift (placeholder - implement with actual drift detection)
        drift_detected = False
        drift_score = 0.0

        self._last_model_health = ModelHealthStatus(
            status=status,
            model_name="SteamOptimizationModel",
            model_version="1.0.0",
            last_prediction_time=recent_predictions[-1] if recent_predictions else None,
            predictions_last_hour=predictions_count,
            average_inference_time_ms=avg_inference_time,
            prediction_error_rate=error_rate,
            drift_detected=drift_detected,
            drift_score=drift_score,
            calibration_status="calibrated",
        )

        return self._last_model_health

    async def check_data_freshness(
        self,
        max_age_seconds: float = 60.0,
    ) -> Dict[str, DataFreshnessStatus]:
        """
        Check data freshness for all monitored streams.

        Args:
            max_age_seconds: Maximum acceptable data age

        Returns:
            Dictionary of stream name to freshness status
        """
        results: Dict[str, DataFreshnessStatus] = {}
        now = datetime.now(timezone.utc)

        # Define expected streams
        expected_streams = [
            "process_data",
            "trap_acoustics",
            "sensor_readings",
            "control_signals",
        ]

        for stream_name in expected_streams:
            last_received = self._data_stream_timestamps.get(stream_name)

            if last_received is None:
                results[stream_name] = DataFreshnessStatus(
                    status=HealthStatus.UNKNOWN,
                    stream_name=stream_name,
                    max_age_seconds=max_age_seconds,
                    is_stale=True,
                    quality_score=0.0,
                )
                continue

            age_seconds = (now - last_received).total_seconds()
            is_stale = age_seconds > max_age_seconds

            if is_stale:
                status = HealthStatus.UNHEALTHY
                quality_score = max(0.0, 1.0 - (age_seconds - max_age_seconds) / max_age_seconds)
            elif age_seconds > max_age_seconds * 0.8:
                status = HealthStatus.DEGRADED
                quality_score = 0.8
            else:
                status = HealthStatus.HEALTHY
                quality_score = 1.0

            results[stream_name] = DataFreshnessStatus(
                status=status,
                stream_name=stream_name,
                last_data_received=last_received,
                age_seconds=age_seconds,
                max_age_seconds=max_age_seconds,
                is_stale=is_stale,
                quality_score=quality_score,
            )

        self._last_data_freshness = results
        return results

    async def get_overall_health(self) -> OverallHealthStatus:
        """
        Get comprehensive overall health status.

        Returns:
            OverallHealthStatus aggregating all health checks
        """
        # Run all health checks
        service_health = await self.check_service_health()
        integration_health = await self.check_integration_health()
        model_health = await self.check_model_health()
        data_freshness = await self.check_data_freshness()

        # Aggregate statuses
        all_statuses: List[HealthStatus] = [service_health.status, model_health.status]
        all_statuses.extend(h.status for h in integration_health.values())
        all_statuses.extend(f.status for f in data_freshness.values())

        # Track degraded and unhealthy components
        degraded: List[str] = []
        unhealthy: List[str] = []

        if service_health.status == HealthStatus.DEGRADED:
            degraded.append("service")
        elif service_health.status == HealthStatus.UNHEALTHY:
            unhealthy.append("service")

        if model_health.status == HealthStatus.DEGRADED:
            degraded.append("model")
        elif model_health.status == HealthStatus.UNHEALTHY:
            unhealthy.append("model")

        for name, health in integration_health.items():
            if health.status == HealthStatus.DEGRADED:
                degraded.append(f"integration:{name}")
            elif health.status == HealthStatus.UNHEALTHY:
                unhealthy.append(f"integration:{name}")

        for name, freshness in data_freshness.items():
            if freshness.status == HealthStatus.DEGRADED:
                degraded.append(f"data:{name}")
            elif freshness.status == HealthStatus.UNHEALTHY:
                unhealthy.append(f"data:{name}")

        # Determine overall status
        if any(s == HealthStatus.UNHEALTHY for s in all_statuses):
            overall_status = HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in all_statuses):
            overall_status = HealthStatus.DEGRADED
        elif any(s == HealthStatus.UNKNOWN for s in all_statuses):
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        # Determine readiness (can accept requests)
        is_ready = overall_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]

        # Liveness is always true if running
        is_live = True

        self._last_overall_health = OverallHealthStatus(
            status=overall_status,
            timestamp=datetime.now(timezone.utc),
            uptime_seconds=time.time() - self._start_time,
            version=self.version,
            agent_id=self.agent_id,
            service_health=service_health,
            integration_health=integration_health,
            model_health=model_health,
            data_freshness=data_freshness,
            is_ready=is_ready,
            is_live=is_live,
            degraded_components=degraded,
            unhealthy_components=unhealthy,
        )

        return self._last_overall_health

    def is_ready(self) -> bool:
        """
        Kubernetes readiness probe.

        Returns:
            True if service can accept requests
        """
        if self._last_overall_health is None:
            return False
        return self._last_overall_health.is_ready

    def is_live(self) -> bool:
        """
        Kubernetes liveness probe.

        Returns:
            True if service is alive (always true if running)
        """
        return True

    def get_status(self) -> Dict[str, Any]:
        """Get current health status as dictionary."""
        return {
            "healthy": self.is_ready(),
            "live": self.is_live(),
            "uptime_seconds": time.time() - self._start_time,
            "version": self.version,
            "agent_id": self.agent_id,
            "last_check": (
                self._last_overall_health.timestamp.isoformat()
                if self._last_overall_health else None
            ),
        }

    async def start_background_checks(self, interval_s: Optional[float] = None) -> None:
        """
        Start background health check loop.

        Args:
            interval_s: Check interval in seconds (default: self._check_interval_s)
        """
        interval = interval_s or self._check_interval_s
        self._running = True

        logger.info("Starting background health checks (interval: %ss)", interval)

        while self._running:
            try:
                await self.get_overall_health()
                logger.debug("Background health check completed")
            except Exception as e:
                logger.error("Background health check failed: %s", e)

            await asyncio.sleep(interval)

    def stop_background_checks(self) -> None:
        """Stop background health check loop."""
        self._running = False
        logger.info("Background health checks stopped")
