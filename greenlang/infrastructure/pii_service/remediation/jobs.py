# -*- coding: utf-8 -*-
"""
PII Remediation Scheduled Jobs - SEC-011 PII Detection/Redaction Enhancements

Scheduled job management for automated PII remediation. Provides background
processing of pending remediation items with configurable intervals, health
monitoring, and graceful shutdown.

Features:
    - Configurable processing intervals
    - Health check endpoint support
    - Graceful shutdown handling
    - Prometheus metrics for job monitoring
    - PushGateway integration for batch job metrics (OBS-001)
    - Integration with Kubernetes CronJobs

Usage:
    >>> from greenlang.infrastructure.pii_service.remediation import (
    ...     PIIRemediationJob,
    ...     PIIRemediationEngine,
    ... )
    >>> engine = PIIRemediationEngine(config)
    >>> job = PIIRemediationJob(engine, interval_minutes=60)
    >>> await job.start()  # Runs until stopped
    >>> await job.stop()

Author: GreenLang Framework Team
Date: February 2026
PRD: SEC-011 PII Detection/Redaction Enhancements, OBS-001 Phase 3
Status: Production Ready
"""

from __future__ import annotations

import asyncio
import logging
import signal
from datetime import datetime, timedelta
from typing import Any, Callable, List, Optional

from pydantic import BaseModel, Field

from greenlang.infrastructure.pii_service.remediation.engine import (
    PIIRemediationEngine,
    RemediationConfig,
)
from greenlang.infrastructure.pii_service.remediation.policies import (
    RemediationResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PushGateway Integration (OBS-001 Phase 3)
# ---------------------------------------------------------------------------

try:
    from greenlang.monitoring.pushgateway import (
        BatchJobMetrics,
        get_pushgateway_client,
    )
    _PUSHGATEWAY_AVAILABLE = True
except ImportError:
    _PUSHGATEWAY_AVAILABLE = False
    BatchJobMetrics = None  # type: ignore[assignment, misc]
    get_pushgateway_client = None  # type: ignore[assignment]
    logger.debug("PushGateway SDK not available; batch metrics disabled")


# ---------------------------------------------------------------------------
# Job Configuration
# ---------------------------------------------------------------------------


class JobConfig(BaseModel):
    """Configuration for PIIRemediationJob.

    Attributes:
        interval_minutes: Time between processing runs.
        max_consecutive_failures: Max failures before stopping.
        failure_backoff_minutes: Backoff time after failure.
        enable_health_check: Enable health check endpoint.
        health_check_port: Port for health check HTTP server.
        graceful_shutdown_timeout: Timeout for graceful shutdown.
        enable_metrics: Emit Prometheus metrics.
        run_on_start: Run immediately on start before first interval.
    """

    interval_minutes: int = Field(
        default=60,
        ge=1,
        le=1440,  # Max 24 hours
        description="Minutes between processing runs"
    )
    max_consecutive_failures: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Max consecutive failures before stopping"
    )
    failure_backoff_minutes: int = Field(
        default=5,
        ge=1,
        le=60,
        description="Backoff time after failure"
    )
    enable_health_check: bool = Field(
        default=True,
        description="Enable health check endpoint"
    )
    health_check_port: int = Field(
        default=8081,
        ge=1024,
        le=65535,
        description="Health check HTTP port"
    )
    graceful_shutdown_timeout: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Graceful shutdown timeout in seconds"
    )
    enable_metrics: bool = Field(
        default=True,
        description="Emit Prometheus metrics"
    )
    run_on_start: bool = Field(
        default=False,
        description="Run immediately on start"
    )

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Job Status
# ---------------------------------------------------------------------------


class JobStatus(BaseModel):
    """Current status of the remediation job.

    Attributes:
        running: Whether the job is currently running.
        last_run_at: Timestamp of last processing run.
        next_run_at: Timestamp of next scheduled run.
        consecutive_failures: Number of consecutive failures.
        total_runs: Total number of runs since start.
        total_processed: Total items processed since start.
        total_failed: Total items failed since start.
        started_at: When the job was started.
        healthy: Whether the job is healthy.
    """

    running: bool = Field(default=False, description="Job is running")
    last_run_at: Optional[datetime] = Field(default=None, description="Last run timestamp")
    next_run_at: Optional[datetime] = Field(default=None, description="Next run timestamp")
    consecutive_failures: int = Field(default=0, description="Consecutive failures")
    total_runs: int = Field(default=0, description="Total runs")
    total_processed: int = Field(default=0, description="Total items processed")
    total_failed: int = Field(default=0, description="Total items failed")
    started_at: Optional[datetime] = Field(default=None, description="Start timestamp")
    healthy: bool = Field(default=True, description="Job is healthy")

    def is_healthy(self, max_failures: int) -> bool:
        """Check if job is in healthy state.

        Args:
            max_failures: Maximum allowed consecutive failures.

        Returns:
            True if healthy.
        """
        return (
            self.running
            and self.consecutive_failures < max_failures
        )


# ---------------------------------------------------------------------------
# Metrics (lazy initialization)
# ---------------------------------------------------------------------------

_metrics_initialized = False
_job_runs_total = None
_job_duration_seconds = None
_job_items_processed_total = None
_job_failures_total = None
_job_healthy = None


def _init_metrics() -> None:
    """Initialize Prometheus metrics lazily."""
    global _metrics_initialized, _job_runs_total, _job_duration_seconds
    global _job_items_processed_total, _job_failures_total, _job_healthy

    if _metrics_initialized:
        return

    try:
        from prometheus_client import Counter, Gauge, Histogram

        _job_runs_total = Counter(
            "gl_pii_remediation_job_runs_total",
            "Total remediation job runs",
            ["status"]
        )
        _job_duration_seconds = Histogram(
            "gl_pii_remediation_job_duration_seconds",
            "Remediation job duration",
            buckets=[1, 5, 10, 30, 60, 120, 300, 600]
        )
        _job_items_processed_total = Counter(
            "gl_pii_remediation_job_items_total",
            "Total items processed by job",
            ["status"]
        )
        _job_failures_total = Counter(
            "gl_pii_remediation_job_failures_total",
            "Total job failures"
        )
        _job_healthy = Gauge(
            "gl_pii_remediation_job_healthy",
            "Whether the job is healthy (1) or not (0)"
        )
        _metrics_initialized = True
    except ImportError:
        logger.debug("prometheus_client not available, metrics disabled")
        _metrics_initialized = True


# ---------------------------------------------------------------------------
# Remediation Job
# ---------------------------------------------------------------------------


class PIIRemediationJob:
    """Scheduled job for PII remediation.

    Runs the remediation engine on a configurable interval, processing
    pending items according to their policies.

    Attributes:
        engine: The remediation engine to use.
        config: Job configuration.

    Example:
        >>> engine = PIIRemediationEngine(engine_config)
        >>> job = PIIRemediationJob(engine, interval_minutes=60)
        >>> await job.start()  # Runs until stopped
    """

    def __init__(
        self,
        engine: PIIRemediationEngine,
        config: Optional[JobConfig] = None,
        interval_minutes: Optional[int] = None,
        enable_pushgateway: bool = True,
    ) -> None:
        """Initialize PIIRemediationJob.

        Args:
            engine: The remediation engine to use.
            config: Job configuration.
            interval_minutes: Override for interval (deprecated, use config).
            enable_pushgateway: Enable PushGateway batch job metrics (OBS-001).
        """
        self._engine = engine

        # Handle legacy interval_minutes parameter
        if config is None:
            if interval_minutes is not None:
                config = JobConfig(interval_minutes=interval_minutes)
            else:
                config = JobConfig()

        self._config = config
        self._status = JobStatus()
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._health_server: Optional[Any] = None
        self._callbacks: List[Callable[[RemediationResult], None]] = []

        # PushGateway batch job metrics (OBS-001 Phase 3)
        self._enable_pushgateway = enable_pushgateway and _PUSHGATEWAY_AVAILABLE
        self._pushgateway_metrics: Optional[BatchJobMetrics] = None
        if self._enable_pushgateway:
            self._pushgateway_metrics = get_pushgateway_client(
                "pii-remediation-job",
            )
            logger.info("PushGateway batch metrics enabled for PIIRemediationJob")

        if self._config.enable_metrics:
            _init_metrics()

    @property
    def status(self) -> JobStatus:
        """Get current job status."""
        return self._status

    @property
    def is_running(self) -> bool:
        """Check if job is running."""
        return self._running

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        """Start the remediation job.

        Runs continuously until stop() is called or max failures reached.
        """
        if self._running:
            logger.warning("Job already running")
            return

        logger.info(
            "Starting PIIRemediationJob with interval=%d minutes",
            self._config.interval_minutes
        )

        self._running = True
        self._status.running = True
        self._status.started_at = datetime.utcnow()
        self._status.consecutive_failures = 0
        self._shutdown_event.clear()

        # Update healthy metric
        if _job_healthy:
            _job_healthy.set(1)

        # Start health check server
        if self._config.enable_health_check:
            await self._start_health_server()

        # Register signal handlers
        self._register_signal_handlers()

        # Run on start if configured
        if self._config.run_on_start:
            await self._execute_run()

        # Main loop
        try:
            while self._running:
                # Calculate next run
                self._status.next_run_at = (
                    datetime.utcnow()
                    + timedelta(minutes=self._config.interval_minutes)
                )

                # Wait for interval or shutdown
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self._config.interval_minutes * 60
                    )
                    # Shutdown requested
                    break
                except asyncio.TimeoutError:
                    # Normal timeout, run processing
                    pass

                if not self._running:
                    break

                await self._execute_run()

                # Check if we've hit max failures
                if self._status.consecutive_failures >= self._config.max_consecutive_failures:
                    logger.error(
                        "Max consecutive failures (%d) reached, stopping job",
                        self._config.max_consecutive_failures
                    )
                    break

        except asyncio.CancelledError:
            logger.info("Job cancelled")
        finally:
            await self._cleanup()

    async def stop(self) -> None:
        """Stop the remediation job gracefully."""
        if not self._running:
            return

        logger.info("Stopping PIIRemediationJob")
        self._running = False
        self._shutdown_event.set()

        # Wait for current run to complete
        if self._task and not self._task.done():
            try:
                await asyncio.wait_for(
                    self._task,
                    timeout=self._config.graceful_shutdown_timeout
                )
            except asyncio.TimeoutError:
                logger.warning("Graceful shutdown timeout, forcing stop")
                self._task.cancel()

        await self._cleanup()

    async def run_once(self) -> RemediationResult:
        """Run a single remediation cycle.

        Useful for testing or manual triggering.

        Returns:
            RemediationResult from the run.
        """
        logger.info("Running single remediation cycle")
        return await self._execute_run()

    # -------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------

    async def _execute_run(self) -> RemediationResult:
        """Execute a single remediation run.

        Returns:
            RemediationResult from the engine.
        """
        start_time = datetime.utcnow()
        self._status.last_run_at = start_time
        self._status.total_runs += 1

        # Start PushGateway tracking (OBS-001 Phase 3)
        if self._pushgateway_metrics:
            self._pushgateway_metrics.set_status("running")

        try:
            result = await self._engine.process_pending_remediations()

            # Update status
            self._status.total_processed += result.processed
            self._status.total_failed += result.failed
            self._status.consecutive_failures = 0
            self._status.healthy = True

            # Update metrics
            if _job_runs_total:
                _job_runs_total.labels(status="success").inc()
            if _job_items_processed_total:
                _job_items_processed_total.labels(status="success").inc(result.processed)
                _job_items_processed_total.labels(status="failed").inc(result.failed)
            if _job_healthy:
                _job_healthy.set(1)

            # Record duration
            duration = (datetime.utcnow() - start_time).total_seconds()
            if _job_duration_seconds:
                _job_duration_seconds.observe(duration)

            # Update PushGateway batch metrics (OBS-001 Phase 3)
            if self._pushgateway_metrics:
                self._pushgateway_metrics._record_duration(duration)
                self._pushgateway_metrics.record_records(
                    result.processed, "remediated"
                )
                if result.failed > 0:
                    self._pushgateway_metrics.record_failed_records(
                        result.failed, "remediation_failed"
                    )
                self._pushgateway_metrics.record_success()
                self._pushgateway_metrics.push()

            logger.info(
                "Remediation run complete: processed=%d, failed=%d, duration=%.2fs",
                result.processed,
                result.failed,
                duration
            )

            # Execute callbacks
            for callback in self._callbacks:
                try:
                    callback(result)
                except Exception as e:
                    logger.error("Callback error: %s", e)

            return result

        except Exception as e:
            self._status.consecutive_failures += 1
            self._status.healthy = False

            if _job_runs_total:
                _job_runs_total.labels(status="error").inc()
            if _job_failures_total:
                _job_failures_total.inc()
            if _job_healthy:
                _job_healthy.set(0)

            # Update PushGateway batch metrics on failure (OBS-001 Phase 3)
            if self._pushgateway_metrics:
                duration = (datetime.utcnow() - start_time).total_seconds()
                self._pushgateway_metrics._record_duration(duration)
                self._pushgateway_metrics.record_failure(type(e).__name__)
                self._pushgateway_metrics.push()

            logger.error(
                "Remediation run failed (attempt %d/%d): %s",
                self._status.consecutive_failures,
                self._config.max_consecutive_failures,
                e,
                exc_info=True
            )

            # Apply backoff
            backoff_seconds = (
                self._config.failure_backoff_minutes * 60
                * self._status.consecutive_failures
            )
            logger.info("Applying backoff: %d seconds", backoff_seconds)
            await asyncio.sleep(min(backoff_seconds, 3600))

            return RemediationResult(
                failed=1,
                errors=[str(e)]
            )

    # -------------------------------------------------------------------------
    # Health Check
    # -------------------------------------------------------------------------

    async def _start_health_server(self) -> None:
        """Start the health check HTTP server."""
        try:
            from aiohttp import web

            app = web.Application()
            app.router.add_get("/health", self._health_handler)
            app.router.add_get("/ready", self._ready_handler)
            app.router.add_get("/status", self._status_handler)

            runner = web.AppRunner(app)
            await runner.setup()
            self._health_server = web.TCPSite(
                runner,
                "0.0.0.0",
                self._config.health_check_port
            )
            await self._health_server.start()

            logger.info(
                "Health check server started on port %d",
                self._config.health_check_port
            )

        except ImportError:
            logger.debug("aiohttp not available, health server disabled")
        except Exception as e:
            logger.warning("Failed to start health server: %s", e)

    async def _health_handler(self, request: Any) -> Any:
        """Handle /health endpoint."""
        from aiohttp import web

        if self._status.is_healthy(self._config.max_consecutive_failures):
            return web.json_response({"status": "healthy"}, status=200)
        return web.json_response({"status": "unhealthy"}, status=503)

    async def _ready_handler(self, request: Any) -> Any:
        """Handle /ready endpoint."""
        from aiohttp import web

        if self._running:
            return web.json_response({"status": "ready"}, status=200)
        return web.json_response({"status": "not_ready"}, status=503)

    async def _status_handler(self, request: Any) -> Any:
        """Handle /status endpoint."""
        from aiohttp import web

        return web.json_response({
            "running": self._status.running,
            "healthy": self._status.healthy,
            "last_run_at": (
                self._status.last_run_at.isoformat()
                if self._status.last_run_at else None
            ),
            "next_run_at": (
                self._status.next_run_at.isoformat()
                if self._status.next_run_at else None
            ),
            "consecutive_failures": self._status.consecutive_failures,
            "total_runs": self._status.total_runs,
            "total_processed": self._status.total_processed,
            "total_failed": self._status.total_failed,
        })

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def add_callback(
        self,
        callback: Callable[[RemediationResult], None]
    ) -> None:
        """Add a callback to be called after each run.

        Args:
            callback: Function to call with RemediationResult.
        """
        self._callbacks.append(callback)

    def remove_callback(
        self,
        callback: Callable[[RemediationResult], None]
    ) -> None:
        """Remove a callback.

        Args:
            callback: Function to remove.
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    # -------------------------------------------------------------------------
    # Signal Handling
    # -------------------------------------------------------------------------

    def _register_signal_handlers(self) -> None:
        """Register OS signal handlers for graceful shutdown."""
        try:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(
                    sig,
                    lambda: asyncio.create_task(self.stop())
                )
            logger.debug("Signal handlers registered")
        except (NotImplementedError, RuntimeError):
            # Signal handlers not supported (e.g., Windows)
            logger.debug("Signal handlers not supported on this platform")

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    async def _cleanup(self) -> None:
        """Clean up resources."""
        self._running = False
        self._status.running = False

        # Stop health server
        if self._health_server:
            await self._health_server.stop()
            self._health_server = None

        # Update metric
        if _job_healthy:
            _job_healthy.set(0)

        logger.info("PIIRemediationJob stopped")


# ---------------------------------------------------------------------------
# Kubernetes CronJob Runner
# ---------------------------------------------------------------------------


async def run_remediation_cron(
    engine_config: Optional[RemediationConfig] = None,
    audit_service: Optional[Any] = None,
    db_pool: Optional[Any] = None,
    enable_pushgateway: bool = True,
) -> RemediationResult:
    """Run remediation as a one-shot job.

    Designed for Kubernetes CronJob execution. Runs a single remediation
    cycle and exits. Integrates with PushGateway for batch job metrics
    when running as a CronJob (OBS-001 Phase 3).

    Args:
        engine_config: Engine configuration.
        audit_service: Audit service for logging.
        db_pool: Database connection pool.
        enable_pushgateway: Enable PushGateway batch job metrics.

    Returns:
        RemediationResult from the run.

    Example:
        >>> # In a K8s CronJob container
        >>> result = await run_remediation_cron()
        >>> sys.exit(0 if result.failed == 0 else 1)
    """
    logger.info("Running PII remediation cron job")

    # Initialize PushGateway metrics for CronJob (OBS-001 Phase 3)
    pushgateway_metrics: Optional[BatchJobMetrics] = None
    if enable_pushgateway and _PUSHGATEWAY_AVAILABLE:
        pushgateway_metrics = get_pushgateway_client("pii-remediation-cron")
        logger.info("PushGateway batch metrics enabled for remediation cron")

    engine = PIIRemediationEngine(
        config=engine_config,
        audit_service=audit_service,
        db_pool=db_pool,
    )
    await engine.initialize()

    try:
        # Track duration with PushGateway
        if pushgateway_metrics:
            pushgateway_metrics.set_status("running")

        start_time = datetime.utcnow()
        result = await engine.process_pending_remediations()
        duration = (datetime.utcnow() - start_time).total_seconds()

        # Update PushGateway metrics (OBS-001 Phase 3)
        if pushgateway_metrics:
            pushgateway_metrics._record_duration(duration)
            pushgateway_metrics.record_records(result.processed, "remediated")
            if result.failed > 0:
                pushgateway_metrics.record_failed_records(
                    result.failed, "remediation_failed"
                )
            pushgateway_metrics.record_success()
            pushgateway_metrics.push()

        logger.info(
            "Cron job complete: processed=%d, failed=%d, duration=%.2fs",
            result.processed,
            result.failed,
            duration,
        )

        return result

    except Exception as e:
        # Record failure in PushGateway (OBS-001 Phase 3)
        if pushgateway_metrics:
            pushgateway_metrics.record_failure(type(e).__name__)
            pushgateway_metrics.push()
        raise

    finally:
        await engine.close()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "JobConfig",
    "JobStatus",
    "PIIRemediationJob",
    "run_remediation_cron",
]
