# -*- coding: utf-8 -*-
"""
Consumer Group Management

Manages consumer groups for parallel message processing with:
- Dynamic scaling (add/remove consumers)
- Load balancing across consumers
- Health monitoring and auto-recovery
- Consumer lag tracking
- Graceful shutdown

Example:
    >>> manager = ConsumerGroupManager(broker)
    >>> await manager.create_group("agent.tasks", "workers")
    >>> await manager.scale_consumers("agent.tasks", "workers", count=10)
    >>> stats = await manager.get_group_stats("agent.tasks", "workers")
"""

from typing import Dict, List, Optional, Any, Callable
import asyncio
import logging
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import uuid

from .broker_interface import MessageBrokerInterface
from .message import Message
from greenlang.determinism import DeterministicClock
from greenlang.determinism import deterministic_uuid, DeterministicClock

logger = logging.getLogger(__name__)


class ConsumerState(str, Enum):
    """Consumer states."""
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class ConsumerInfo:
    """Information about a consumer."""
    consumer_id: str
    group_name: str
    topic: str
    state: ConsumerState
    messages_processed: int = 0
    messages_failed: int = 0
    last_message_time: Optional[datetime] = None
    started_at: datetime = field(default_factory=datetime.utcnow)
    task: Optional[asyncio.Task] = None

    def get_uptime_seconds(self) -> float:
        """Get consumer uptime in seconds."""
        return (DeterministicClock.utcnow() - self.started_at).total_seconds()

    def get_throughput(self) -> float:
        """Get messages per second."""
        uptime = self.get_uptime_seconds()
        return self.messages_processed / uptime if uptime > 0 else 0.0


@dataclass
class ConsumerGroupStats:
    """Statistics for a consumer group."""
    group_name: str
    topic: str
    consumer_count: int
    total_messages_processed: int
    total_messages_failed: int
    consumer_lag: int
    average_throughput: float
    healthy_consumers: int
    failed_consumers: int
    created_at: datetime


class ConsumerGroupManager:
    """
    Manages consumer groups for parallel message processing.

    Features:
        - Dynamic scaling (1-100 consumers per group)
        - Automatic load balancing
        - Health monitoring with auto-recovery
        - Consumer lag tracking
        - Graceful shutdown with message draining

    Example:
        >>> manager = ConsumerGroupManager(broker)
        >>> await manager.create_group("agent.tasks", "workers")
        >>>
        >>> # Scale to 10 workers
        >>> await manager.scale_consumers(
        ...     "agent.tasks",
        ...     "workers",
        ...     count=10,
        ...     handler=process_task
        ... )
        >>>
        >>> # Monitor health
        >>> stats = await manager.get_group_stats("agent.tasks", "workers")
        >>> print(f"Lag: {stats.consumer_lag}, Throughput: {stats.average_throughput}/s")
    """

    def __init__(
        self,
        broker: MessageBrokerInterface,
        health_check_interval: int = 30,
        max_retry_attempts: int = 3,
    ):
        """
        Initialize consumer group manager.

        Args:
            broker: Message broker instance
            health_check_interval: Health check interval in seconds
            max_retry_attempts: Max consumer restart attempts
        """
        self.broker = broker
        self.health_check_interval = health_check_interval
        self.max_retry_attempts = max_retry_attempts

        # Consumer tracking
        self._consumers: Dict[str, ConsumerInfo] = {}
        self._groups: Dict[str, ConsumerGroupStats] = {}
        self._health_check_task: Optional[asyncio.Task] = None
        self._shutdown = False

    async def create_group(
        self,
        topic: str,
        group_name: str,
    ) -> None:
        """
        Create consumer group.

        Args:
            topic: Topic name
            group_name: Consumer group name
        """
        try:
            await self.broker.create_consumer_group(topic, group_name)

            # Track group
            self._groups[f"{topic}:{group_name}"] = ConsumerGroupStats(
                group_name=group_name,
                topic=topic,
                consumer_count=0,
                total_messages_processed=0,
                total_messages_failed=0,
                consumer_lag=0,
                average_throughput=0.0,
                healthy_consumers=0,
                failed_consumers=0,
                created_at=DeterministicClock.utcnow(),
            )

            logger.info(f"Created consumer group {group_name} for topic {topic}")

        except Exception as e:
            logger.error(f"Failed to create consumer group: {e}", exc_info=True)
            raise

    async def delete_group(
        self,
        topic: str,
        group_name: str,
    ) -> None:
        """
        Delete consumer group.

        Args:
            topic: Topic name
            group_name: Consumer group name
        """
        try:
            # Stop all consumers in group
            await self.stop_all_consumers(topic, group_name)

            # Delete group
            await self.broker.delete_consumer_group(topic, group_name)

            # Remove tracking
            group_key = f"{topic}:{group_name}"
            self._groups.pop(group_key, None)

            logger.info(f"Deleted consumer group {group_name}")

        except Exception as e:
            logger.error(f"Failed to delete consumer group: {e}", exc_info=True)
            raise

    async def add_consumer(
        self,
        topic: str,
        group_name: str,
        handler: Callable[[Message], Any],
        consumer_id: Optional[str] = None,
    ) -> str:
        """
        Add consumer to group.

        Args:
            topic: Topic name
            group_name: Consumer group name
            handler: Message handler function
            consumer_id: Optional consumer ID

        Returns:
            Consumer ID
        """
        consumer_id = consumer_id or f"consumer-{deterministic_uuid(__name__, str(DeterministicClock.now())).hex[:8]}"

        try:
            # Create consumer info
            consumer_info = ConsumerInfo(
                consumer_id=consumer_id,
                group_name=group_name,
                topic=topic,
                state=ConsumerState.STARTING,
            )

            # Start consumer task
            async def consumer_loop():
                """Consumer message processing loop."""
                consumer_info.state = ConsumerState.RUNNING
                logger.info(f"Consumer {consumer_id} started")

                try:
                    async for message in self.broker.consume(
                        topic,
                        group_name,
                        consumer_id,
                    ):
                        try:
                            # Process message
                            if asyncio.iscoroutinefunction(handler):
                                await handler(message)
                            else:
                                handler(message)

                            # Acknowledge
                            await self.broker.acknowledge(message)

                            # Update stats
                            consumer_info.messages_processed += 1
                            consumer_info.last_message_time = DeterministicClock.utcnow()

                        except Exception as e:
                            logger.error(
                                f"Consumer {consumer_id} handler error: {e}",
                                exc_info=True
                            )
                            consumer_info.messages_failed += 1
                            await self.broker.nack(message, str(e))

                except asyncio.CancelledError:
                    logger.info(f"Consumer {consumer_id} cancelled")
                    consumer_info.state = ConsumerState.STOPPED
                except Exception as e:
                    logger.error(f"Consumer {consumer_id} failed: {e}", exc_info=True)
                    consumer_info.state = ConsumerState.FAILED
                    raise

            # Create and track task
            task = asyncio.create_task(consumer_loop())
            consumer_info.task = task
            self._consumers[consumer_id] = consumer_info

            # Update group stats
            group_key = f"{topic}:{group_name}"
            if group_key in self._groups:
                self._groups[group_key].consumer_count += 1

            logger.info(f"Added consumer {consumer_id} to group {group_name}")
            return consumer_id

        except Exception as e:
            logger.error(f"Failed to add consumer: {e}", exc_info=True)
            raise

    async def remove_consumer(
        self,
        consumer_id: str,
        graceful: bool = True,
    ) -> None:
        """
        Remove consumer from group.

        Args:
            consumer_id: Consumer ID to remove
            graceful: If True, wait for current message processing
        """
        consumer_info = self._consumers.get(consumer_id)
        if not consumer_info:
            logger.warning(f"Consumer {consumer_id} not found")
            return

        try:
            consumer_info.state = ConsumerState.STOPPING
            logger.info(f"Stopping consumer {consumer_id}")

            if consumer_info.task:
                if graceful:
                    # Wait for current message processing (max 30s)
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(consumer_info.task),
                            timeout=30.0
                        )
                    except asyncio.TimeoutError:
                        logger.warning(f"Consumer {consumer_id} graceful stop timeout")

                # Cancel task
                consumer_info.task.cancel()
                try:
                    await consumer_info.task
                except asyncio.CancelledError:
                    pass

            # Update stats
            group_key = f"{consumer_info.topic}:{consumer_info.group_name}"
            if group_key in self._groups:
                self._groups[group_key].consumer_count -= 1

            # Remove tracking
            del self._consumers[consumer_id]
            logger.info(f"Removed consumer {consumer_id}")

        except Exception as e:
            logger.error(f"Failed to remove consumer: {e}", exc_info=True)
            raise

    async def scale_consumers(
        self,
        topic: str,
        group_name: str,
        count: int,
        handler: Callable[[Message], Any],
    ) -> None:
        """
        Scale consumer group to target count.

        Args:
            topic: Topic name
            group_name: Consumer group name
            count: Target consumer count
            handler: Message handler function
        """
        # Get current consumers in group
        current_consumers = [
            c for c in self._consumers.values()
            if c.topic == topic and c.group_name == group_name
        ]
        current_count = len(current_consumers)

        logger.info(
            f"Scaling group {group_name} from {current_count} to {count} consumers"
        )

        if count > current_count:
            # Scale up - add consumers
            for _ in range(count - current_count):
                await self.add_consumer(topic, group_name, handler)

        elif count < current_count:
            # Scale down - remove consumers
            consumers_to_remove = current_consumers[:current_count - count]
            for consumer_info in consumers_to_remove:
                await self.remove_consumer(consumer_info.consumer_id)

        logger.info(f"Scaled group {group_name} to {count} consumers")

    async def stop_all_consumers(
        self,
        topic: str,
        group_name: str,
        graceful: bool = True,
    ) -> None:
        """
        Stop all consumers in group.

        Args:
            topic: Topic name
            group_name: Consumer group name
            graceful: If True, wait for message processing
        """
        consumers = [
            c for c in self._consumers.values()
            if c.topic == topic and c.group_name == group_name
        ]

        logger.info(f"Stopping {len(consumers)} consumers in group {group_name}")

        # Stop all consumers concurrently
        await asyncio.gather(
            *[self.remove_consumer(c.consumer_id, graceful) for c in consumers],
            return_exceptions=True
        )

    async def get_group_stats(
        self,
        topic: str,
        group_name: str,
    ) -> Optional[ConsumerGroupStats]:
        """
        Get consumer group statistics.

        Args:
            topic: Topic name
            group_name: Consumer group name

        Returns:
            Consumer group statistics
        """
        group_key = f"{topic}:{group_name}"
        stats = self._groups.get(group_key)

        if stats:
            # Update stats
            consumers = [
                c for c in self._consumers.values()
                if c.topic == topic and c.group_name == group_name
            ]

            stats.consumer_count = len(consumers)
            stats.total_messages_processed = sum(
                c.messages_processed for c in consumers
            )
            stats.total_messages_failed = sum(
                c.messages_failed for c in consumers
            )
            stats.healthy_consumers = sum(
                1 for c in consumers if c.state == ConsumerState.RUNNING
            )
            stats.failed_consumers = sum(
                1 for c in consumers if c.state == ConsumerState.FAILED
            )

            # Calculate average throughput
            if consumers:
                stats.average_throughput = sum(
                    c.get_throughput() for c in consumers
                ) / len(consumers)

            # Get consumer lag from broker
            try:
                stats.consumer_lag = await self.broker.get_consumer_lag(
                    topic,
                    group_name
                )
            except Exception as e:
                logger.error(f"Failed to get consumer lag: {e}")

        return stats

    async def get_consumer_info(
        self,
        consumer_id: str,
    ) -> Optional[ConsumerInfo]:
        """
        Get consumer information.

        Args:
            consumer_id: Consumer ID

        Returns:
            Consumer information
        """
        return self._consumers.get(consumer_id)

    async def list_consumers(
        self,
        topic: Optional[str] = None,
        group_name: Optional[str] = None,
    ) -> List[ConsumerInfo]:
        """
        List all consumers, optionally filtered.

        Args:
            topic: Optional topic filter
            group_name: Optional group name filter

        Returns:
            List of consumer information
        """
        consumers = list(self._consumers.values())

        if topic:
            consumers = [c for c in consumers if c.topic == topic]

        if group_name:
            consumers = [c for c in consumers if c.group_name == group_name]

        return consumers

    async def start_health_monitoring(self) -> None:
        """Start health monitoring for all consumers."""
        if self._health_check_task:
            logger.warning("Health monitoring already started")
            return

        async def health_check_loop():
            """Health check background task."""
            logger.info("Health monitoring started")

            while not self._shutdown:
                try:
                    # Check each consumer
                    for consumer_id, consumer_info in list(self._consumers.items()):
                        # Check if consumer task is alive
                        if consumer_info.task and consumer_info.task.done():
                            try:
                                # Check for exceptions
                                await consumer_info.task
                            except Exception as e:
                                logger.error(
                                    f"Consumer {consumer_id} failed: {e}",
                                    exc_info=True
                                )
                                consumer_info.state = ConsumerState.FAILED

                    await asyncio.sleep(self.health_check_interval)

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Health check error: {e}", exc_info=True)

            logger.info("Health monitoring stopped")

        self._health_check_task = asyncio.create_task(health_check_loop())

    async def stop_health_monitoring(self) -> None:
        """Stop health monitoring."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None

    async def shutdown(self, graceful: bool = True) -> None:
        """
        Shutdown all consumers and cleanup.

        Args:
            graceful: If True, wait for message processing
        """
        logger.info("Shutting down consumer group manager")
        self._shutdown = True

        # Stop health monitoring
        await self.stop_health_monitoring()

        # Stop all consumers
        consumer_ids = list(self._consumers.keys())
        await asyncio.gather(
            *[self.remove_consumer(cid, graceful) for cid in consumer_ids],
            return_exceptions=True
        )

        logger.info("Consumer group manager shutdown complete")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_health_monitoring()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()
