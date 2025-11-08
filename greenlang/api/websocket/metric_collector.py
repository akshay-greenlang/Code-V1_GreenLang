"""
Background task for collecting and publishing metrics.

This module collects system, workflow, agent, and distributed metrics
and publishes them to Redis pub/sub channels for real-time streaming.
"""

import asyncio
import logging
import platform
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import msgpack
import psutil
import redis.asyncio as aioredis
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

# Configure logging
logger = logging.getLogger(__name__)


class MetricRetentionPolicy(BaseModel):
    """Retention policy for metrics."""

    raw_retention: timedelta = Field(
        default=timedelta(hours=1),
        description="Retention for raw 1s metrics"
    )
    short_retention: timedelta = Field(
        default=timedelta(hours=24),
        description="Retention for 1m aggregated metrics"
    )
    medium_retention: timedelta = Field(
        default=timedelta(days=7),
        description="Retention for 5m aggregated metrics"
    )
    long_retention: timedelta = Field(
        default=timedelta(days=30),
        description="Retention for 1h aggregated metrics"
    )
    archive_retention: timedelta = Field(
        default=timedelta(days=365),
        description="Retention for 1d aggregated metrics"
    )


class SystemMetrics:
    """Collect system metrics using psutil."""

    def __init__(self):
        """Initialize system metrics collector."""
        self.cpu_percent_interval = 1.0
        self.disk_partitions = psutil.disk_partitions()

    def collect(self) -> Dict[str, Any]:
        """Collect current system metrics.

        Returns:
            Dictionary of system metrics
        """
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None, percpu=True)
            cpu_count = psutil.cpu_count(logical=True)
            cpu_freq = psutil.cpu_freq()

            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()

            # Disk metrics
            disk_io = psutil.disk_io_counters()
            disk_usage = {}
            for partition in self.disk_partitions:
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_usage[partition.mountpoint] = {
                        "total": usage.total,
                        "used": usage.used,
                        "free": usage.free,
                        "percent": usage.percent
                    }
                except (PermissionError, OSError):
                    pass

            # Network metrics
            network = psutil.net_io_counters()

            # Process metrics
            process = psutil.Process()
            process_info = {
                "cpu_percent": process.cpu_percent(interval=None),
                "memory_percent": process.memory_percent(),
                "memory_rss": process.memory_info().rss,
                "num_threads": process.num_threads(),
                "num_fds": process.num_fds() if hasattr(process, "num_fds") else 0
            }

            return {
                "timestamp": datetime.utcnow().isoformat(),
                "cpu": {
                    "percent": sum(cpu_percent) / len(cpu_percent),
                    "percent_per_cpu": cpu_percent,
                    "count": cpu_count,
                    "frequency": cpu_freq.current if cpu_freq else 0
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "used": memory.used,
                    "percent": memory.percent,
                    "swap_total": swap.total,
                    "swap_used": swap.used,
                    "swap_percent": swap.percent
                },
                "disk": {
                    "io_read_bytes": disk_io.read_bytes if disk_io else 0,
                    "io_write_bytes": disk_io.write_bytes if disk_io else 0,
                    "io_read_count": disk_io.read_count if disk_io else 0,
                    "io_write_count": disk_io.write_count if disk_io else 0,
                    "usage": disk_usage
                },
                "network": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv,
                    "errors_in": network.errin,
                    "errors_out": network.errout
                },
                "process": process_info,
                "platform": {
                    "system": platform.system(),
                    "release": platform.release(),
                    "version": platform.version(),
                    "machine": platform.machine()
                }
            }
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {}


class WorkflowMetrics:
    """Collect workflow execution metrics from database."""

    def __init__(self, db_session: AsyncSession):
        """Initialize workflow metrics collector.

        Args:
            db_session: Database session
        """
        self.db_session = db_session

    async def collect(self, time_window: int = 60) -> Dict[str, Any]:
        """Collect workflow metrics.

        Args:
            time_window: Time window in seconds

        Returns:
            Dictionary of workflow metrics
        """
        try:
            # This would query actual workflow execution tables
            # For now, return placeholder data
            current_time = datetime.utcnow()
            window_start = current_time - timedelta(seconds=time_window)

            metrics = {
                "timestamp": current_time.isoformat(),
                "time_window": time_window,
                "executions": {
                    "total": 0,
                    "successful": 0,
                    "failed": 0,
                    "running": 0,
                    "pending": 0
                },
                "duration": {
                    "min": 0,
                    "max": 0,
                    "avg": 0,
                    "p50": 0,
                    "p95": 0,
                    "p99": 0
                },
                "success_rate": 0.0,
                "throughput": 0.0  # executions per second
            }

            return metrics
        except Exception as e:
            logger.error(f"Error collecting workflow metrics: {e}")
            return {}


class AgentMetrics:
    """Collect agent execution metrics."""

    def __init__(self, db_session: AsyncSession):
        """Initialize agent metrics collector.

        Args:
            db_session: Database session
        """
        self.db_session = db_session

    async def collect(self, time_window: int = 60) -> Dict[str, Any]:
        """Collect agent metrics.

        Args:
            time_window: Time window in seconds

        Returns:
            Dictionary of agent metrics
        """
        try:
            current_time = datetime.utcnow()
            window_start = current_time - timedelta(seconds=time_window)

            metrics = {
                "timestamp": current_time.isoformat(),
                "time_window": time_window,
                "calls": {
                    "total": 0,
                    "successful": 0,
                    "failed": 0
                },
                "latency": {
                    "min": 0,
                    "max": 0,
                    "avg": 0,
                    "p50": 0,
                    "p95": 0,
                    "p99": 0
                },
                "errors": {
                    "total": 0,
                    "by_type": {}
                },
                "top_agents": [],
                "slowest_agents": []
            }

            return metrics
        except Exception as e:
            logger.error(f"Error collecting agent metrics: {e}")
            return {}


class DistributedMetrics:
    """Collect distributed system metrics from Redis cluster."""

    def __init__(self, redis_client: aioredis.Redis):
        """Initialize distributed metrics collector.

        Args:
            redis_client: Redis client
        """
        self.redis_client = redis_client

    async def collect(self) -> Dict[str, Any]:
        """Collect distributed metrics.

        Returns:
            Dictionary of distributed metrics
        """
        try:
            # Get Redis info
            info = await self.redis_client.info()

            # Get cluster info if available
            cluster_info = {}
            try:
                cluster_info = await self.redis_client.cluster_info()
            except Exception:
                pass

            metrics = {
                "timestamp": datetime.utcnow().isoformat(),
                "redis": {
                    "connected_clients": info.get("connected_clients", 0),
                    "used_memory": info.get("used_memory", 0),
                    "used_memory_rss": info.get("used_memory_rss", 0),
                    "total_commands_processed": info.get("total_commands_processed", 0),
                    "instantaneous_ops_per_sec": info.get("instantaneous_ops_per_sec", 0),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0),
                    "evicted_keys": info.get("evicted_keys", 0)
                },
                "cluster": cluster_info,
                "nodes": [],
                "tasks": {
                    "total": 0,
                    "pending": 0,
                    "running": 0,
                    "completed": 0,
                    "failed": 0
                },
                "throughput": 0.0
            }

            return metrics
        except Exception as e:
            logger.error(f"Error collecting distributed metrics: {e}")
            return {}


class MetricBuffer:
    """Buffer metrics for batching."""

    def __init__(self, max_size: int = 100, max_age: float = 1.0):
        """Initialize metric buffer.

        Args:
            max_size: Maximum buffer size
            max_age: Maximum age in seconds before flush
        """
        self.max_size = max_size
        self.max_age = max_age
        self.buffer: Dict[str, List[Dict[str, Any]]] = {}
        self.last_flush: Dict[str, float] = {}

    def add(self, channel: str, metric: Dict[str, Any]) -> None:
        """Add metric to buffer.

        Args:
            channel: Channel name
            metric: Metric data
        """
        if channel not in self.buffer:
            self.buffer[channel] = []
            self.last_flush[channel] = time.time()

        self.buffer[channel].append(metric)

    def should_flush(self, channel: str) -> bool:
        """Check if channel should be flushed.

        Args:
            channel: Channel name

        Returns:
            True if should flush, False otherwise
        """
        if channel not in self.buffer:
            return False

        buffer_size = len(self.buffer[channel])
        buffer_age = time.time() - self.last_flush.get(channel, 0)

        return buffer_size >= self.max_size or buffer_age >= self.max_age

    def flush(self, channel: str) -> List[Dict[str, Any]]:
        """Flush channel buffer.

        Args:
            channel: Channel name

        Returns:
            List of buffered metrics
        """
        metrics = self.buffer.get(channel, [])
        self.buffer[channel] = []
        self.last_flush[channel] = time.time()
        return metrics


class MetricDownsampler:
    """Downsample high-resolution metrics to lower resolutions."""

    def __init__(self, redis_client: aioredis.Redis):
        """Initialize downsampler.

        Args:
            redis_client: Redis client
        """
        self.redis_client = redis_client

    async def downsample(
        self,
        source_key: str,
        dest_key: str,
        interval: int,
        aggregation: str = "avg"
    ) -> None:
        """Downsample metrics from source to destination.

        Args:
            source_key: Source Redis key pattern
            dest_key: Destination Redis key
            interval: Aggregation interval in seconds
            aggregation: Aggregation function (avg, sum, min, max)
        """
        try:
            # This would implement actual downsampling logic
            # using Redis TimeSeries or similar functionality
            pass
        except Exception as e:
            logger.error(f"Error downsampling metrics: {e}")


class MetricCollector:
    """Background task for collecting and publishing metrics."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        db_session: Optional[AsyncSession] = None,
        collection_interval: int = 1,
        retention_policy: Optional[MetricRetentionPolicy] = None
    ):
        """Initialize metric collector.

        Args:
            redis_url: Redis connection URL
            db_session: Database session for workflow/agent metrics
            collection_interval: Collection interval in seconds
            retention_policy: Metric retention policy
        """
        self.redis_url = redis_url
        self.db_session = db_session
        self.collection_interval = collection_interval
        self.retention_policy = retention_policy or MetricRetentionPolicy()

        self.redis_client: Optional[aioredis.Redis] = None
        self.running = False
        self._tasks: List[asyncio.Task] = []

        # Initialize collectors
        self.system_collector = SystemMetrics()
        self.workflow_collector = WorkflowMetrics(db_session) if db_session else None
        self.agent_collector = AgentMetrics(db_session) if db_session else None
        self.distributed_collector: Optional[DistributedMetrics] = None

        # Initialize buffer
        self.buffer = MetricBuffer()

    async def start(self) -> None:
        """Start metric collection."""
        logger.info("Starting metric collector")

        # Connect to Redis
        self.redis_client = await aioredis.from_url(
            self.redis_url,
            decode_responses=False
        )

        self.distributed_collector = DistributedMetrics(self.redis_client)

        self.running = True

        # Start collection tasks
        self._tasks.append(asyncio.create_task(self._collect_system_metrics()))
        self._tasks.append(asyncio.create_task(self._collect_workflow_metrics()))
        self._tasks.append(asyncio.create_task(self._collect_agent_metrics()))
        self._tasks.append(asyncio.create_task(self._collect_distributed_metrics()))
        self._tasks.append(asyncio.create_task(self._flush_buffer()))
        self._tasks.append(asyncio.create_task(self._cleanup_old_metrics()))

        logger.info("Metric collector started")

    async def stop(self) -> None:
        """Stop metric collection."""
        logger.info("Stopping metric collector")

        self.running = False

        # Cancel tasks
        for task in self._tasks:
            task.cancel()

        await asyncio.gather(*self._tasks, return_exceptions=True)

        # Close Redis
        if self.redis_client:
            await self.redis_client.close()

        logger.info("Metric collector stopped")

    async def _collect_system_metrics(self) -> None:
        """Collect system metrics periodically."""
        while self.running:
            try:
                metrics = self.system_collector.collect()
                if metrics:
                    await self._publish_metric("system.metrics", metrics)

            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")

            await asyncio.sleep(self.collection_interval)

    async def _collect_workflow_metrics(self) -> None:
        """Collect workflow metrics periodically."""
        if not self.workflow_collector:
            return

        while self.running:
            try:
                metrics = await self.workflow_collector.collect()
                if metrics:
                    await self._publish_metric("workflow.metrics", metrics)

            except Exception as e:
                logger.error(f"Error collecting workflow metrics: {e}")

            await asyncio.sleep(5)  # Collect every 5 seconds

    async def _collect_agent_metrics(self) -> None:
        """Collect agent metrics periodically."""
        if not self.agent_collector:
            return

        while self.running:
            try:
                metrics = await self.agent_collector.collect()
                if metrics:
                    await self._publish_metric("agent.metrics", metrics)

            except Exception as e:
                logger.error(f"Error collecting agent metrics: {e}")

            await asyncio.sleep(5)  # Collect every 5 seconds

    async def _collect_distributed_metrics(self) -> None:
        """Collect distributed metrics periodically."""
        if not self.distributed_collector:
            return

        while self.running:
            try:
                metrics = await self.distributed_collector.collect()
                if metrics:
                    await self._publish_metric("distributed.metrics", metrics)

            except Exception as e:
                logger.error(f"Error collecting distributed metrics: {e}")

            await asyncio.sleep(5)  # Collect every 5 seconds

    async def _publish_metric(self, channel: str, metric: Dict[str, Any]) -> None:
        """Publish metric to Redis pub/sub channel.

        Args:
            channel: Channel name
            metric: Metric data
        """
        try:
            # Add to buffer
            self.buffer.add(channel, metric)

            # Flush if needed
            if self.buffer.should_flush(channel):
                await self._flush_channel(channel)

        except Exception as e:
            logger.error(f"Error publishing metric to {channel}: {e}")

    async def _flush_buffer(self) -> None:
        """Flush metric buffer periodically."""
        while self.running:
            await asyncio.sleep(1)

            try:
                for channel in list(self.buffer.buffer.keys()):
                    if self.buffer.should_flush(channel):
                        await self._flush_channel(channel)

            except Exception as e:
                logger.error(f"Error flushing buffer: {e}")

    async def _flush_channel(self, channel: str) -> None:
        """Flush metrics for a channel.

        Args:
            channel: Channel name
        """
        metrics = self.buffer.flush(channel)

        if not metrics:
            return

        try:
            # Publish to Redis pub/sub
            for metric in metrics:
                packed_metric = msgpack.packb(metric, use_bin_type=True)
                await self.redis_client.publish(channel, packed_metric)

            # Store in time series (if using Redis TimeSeries)
            # This would store metrics for historical queries
            await self._store_metrics(channel, metrics)

        except Exception as e:
            logger.error(f"Error flushing channel {channel}: {e}")

    async def _store_metrics(self, channel: str, metrics: List[Dict[str, Any]]) -> None:
        """Store metrics for historical queries.

        Args:
            channel: Channel name
            metrics: List of metrics to store
        """
        try:
            # This would use Redis TimeSeries or similar
            # to store metrics with automatic retention and downsampling
            for metric in metrics:
                key = f"metrics:{channel}:{metric.get('name', 'unknown')}"
                timestamp = int(time.time() * 1000)  # milliseconds

                # Store as hash for now
                await self.redis_client.hset(
                    f"{key}:{timestamp}",
                    mapping={
                        "data": msgpack.packb(metric, use_bin_type=True),
                        "timestamp": timestamp
                    }
                )

                # Set expiration based on retention policy
                await self.redis_client.expire(
                    f"{key}:{timestamp}",
                    int(self.retention_policy.raw_retention.total_seconds())
                )

        except Exception as e:
            logger.error(f"Error storing metrics: {e}")

    async def _cleanup_old_metrics(self) -> None:
        """Clean up old metrics based on retention policy."""
        while self.running:
            # Run cleanup every hour
            await asyncio.sleep(3600)

            try:
                current_time = int(time.time() * 1000)

                # Delete metrics older than retention period
                cutoff_time = current_time - int(
                    self.retention_policy.raw_retention.total_seconds() * 1000
                )

                # This would implement actual cleanup logic
                # using Redis TimeSeries retention or key scanning

            except Exception as e:
                logger.error(f"Error cleaning up old metrics: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get collector statistics.

        Returns:
            Collector statistics
        """
        return {
            "running": self.running,
            "collection_interval": self.collection_interval,
            "buffer_size": sum(len(b) for b in self.buffer.buffer.values()),
            "retention_policy": {
                "raw": self.retention_policy.raw_retention.total_seconds(),
                "short": self.retention_policy.short_retention.total_seconds(),
                "medium": self.retention_policy.medium_retention.total_seconds(),
                "long": self.retention_policy.long_retention.total_seconds(),
                "archive": self.retention_policy.archive_retention.total_seconds()
            }
        }
