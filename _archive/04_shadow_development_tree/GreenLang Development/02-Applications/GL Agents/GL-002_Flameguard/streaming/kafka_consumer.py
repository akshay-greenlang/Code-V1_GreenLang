"""
GL-002 FLAMEGUARD - Kafka Consumer

Consumes boiler events from Kafka topics.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set
import asyncio
import json
import logging

from .event_schemas import (
    BaseEvent,
    ProcessDataEvent,
    OptimizationEvent,
    SafetyEvent,
    EfficiencyEvent,
    EmissionsEvent,
    AlarmEvent,
    EventType,
)

logger = logging.getLogger(__name__)


@dataclass
class ConsumerConfig:
    """Kafka consumer configuration."""
    bootstrap_servers: List[str]
    group_id: str = "flameguard-consumer"
    client_id: str = "flameguard-consumer"

    # Topics to subscribe
    topics: List[str] = None

    # Consumer settings
    auto_offset_reset: str = "latest"
    enable_auto_commit: bool = True
    auto_commit_interval_ms: int = 5000
    max_poll_records: int = 500
    fetch_max_bytes: int = 52428800

    # Processing
    max_concurrent_tasks: int = 10
    processing_timeout_s: float = 30.0

    # Schema Registry
    schema_registry_url: Optional[str] = None
    use_avro: bool = False

    # Security
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: Optional[str] = None
    sasl_username: Optional[str] = None
    sasl_password: Optional[str] = None

    def __post_init__(self):
        if self.topics is None:
            self.topics = [
                "flameguard.process-data",
                "flameguard.optimization",
                "flameguard.safety",
                "flameguard.efficiency",
                "flameguard.emissions",
                "flameguard.alarms",
            ]


class FlameguardKafkaConsumer:
    """
    Kafka consumer for Flameguard events.

    Features:
    - Async message consumption
    - Multiple topic subscription
    - Event handler registration
    - Dead letter queue support
    - Offset management
    """

    def __init__(
        self,
        config: ConsumerConfig,
        on_process_data: Optional[Callable[[ProcessDataEvent], None]] = None,
        on_optimization: Optional[Callable[[OptimizationEvent], None]] = None,
        on_safety: Optional[Callable[[SafetyEvent], None]] = None,
        on_efficiency: Optional[Callable[[EfficiencyEvent], None]] = None,
        on_emissions: Optional[Callable[[EmissionsEvent], None]] = None,
        on_alarm: Optional[Callable[[AlarmEvent], None]] = None,
    ) -> None:
        self.config = config

        # Event handlers
        self._handlers: Dict[EventType, List[Callable]] = {
            EventType.PROCESS_DATA: [on_process_data] if on_process_data else [],
            EventType.OPTIMIZATION: [on_optimization] if on_optimization else [],
            EventType.SAFETY: [on_safety] if on_safety else [],
            EventType.EFFICIENCY: [on_efficiency] if on_efficiency else [],
            EventType.EMISSIONS: [on_emissions] if on_emissions else [],
            EventType.ALARM: [on_alarm] if on_alarm else [],
        }

        # Consumer instance
        self._consumer = None
        self._connected = False
        self._running = False

        # Processing
        self._task_semaphore = asyncio.Semaphore(config.max_concurrent_tasks)
        self._active_tasks: Set[asyncio.Task] = set()

        # Dead letter queue
        self._dlq: List[Dict] = []
        self._max_dlq_size = 1000

        # Statistics
        self._stats = {
            "messages_received": 0,
            "messages_processed": 0,
            "messages_failed": 0,
            "bytes_received": 0,
        }

        logger.info(f"FlameguardKafkaConsumer initialized: {config.bootstrap_servers}")

    def register_handler(
        self,
        event_type: EventType,
        handler: Callable,
    ) -> None:
        """Register event handler."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        logger.debug(f"Registered handler for {event_type}")

    async def connect(self) -> bool:
        """Connect to Kafka cluster."""
        try:
            # In production, use aiokafka:
            # self._consumer = AIOKafkaConsumer(
            #     *self.config.topics,
            #     bootstrap_servers=self.config.bootstrap_servers,
            #     group_id=self.config.group_id,
            #     auto_offset_reset=self.config.auto_offset_reset,
            # )
            # await self._consumer.start()

            self._connected = True
            logger.info(f"Kafka consumer connected, topics: {self.config.topics}")
            return True
        except Exception as e:
            logger.error(f"Kafka connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from Kafka cluster."""
        self._running = False

        # Wait for active tasks
        if self._active_tasks:
            await asyncio.gather(*self._active_tasks, return_exceptions=True)

        if self._consumer:
            # await self._consumer.stop()
            pass
        self._connected = False
        logger.info("Kafka consumer disconnected")

    async def start(self) -> None:
        """Start consuming messages."""
        if not self._connected:
            await self.connect()

        self._running = True
        logger.info("Starting Kafka consumer loop")

        while self._running:
            try:
                # In production:
                # async for msg in self._consumer:
                #     await self._process_message(msg)

                # Simulate message processing
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Consumer error: {e}")
                await asyncio.sleep(1)

    async def stop(self) -> None:
        """Stop consuming messages."""
        self._running = False
        logger.info("Stopping Kafka consumer")

    async def _process_message(self, message: Dict[str, Any]) -> None:
        """Process a single message."""
        async with self._task_semaphore:
            try:
                self._stats["messages_received"] += 1

                # Parse message
                topic = message.get("topic", "")
                key = message.get("key", "")
                value = message.get("value", {})

                if isinstance(value, (str, bytes)):
                    value = json.loads(value)

                self._stats["bytes_received"] += len(str(value))

                # Determine event type
                event_type = self._get_event_type(topic, value)
                if not event_type:
                    logger.warning(f"Unknown event type for topic: {topic}")
                    return

                # Deserialize event
                event = self._deserialize_event(event_type, value)
                if not event:
                    await self._send_to_dlq(message, "Deserialization failed")
                    return

                # Call handlers
                handlers = self._handlers.get(event_type, [])
                for handler in handlers:
                    if handler:
                        try:
                            if asyncio.iscoroutinefunction(handler):
                                await handler(event)
                            else:
                                handler(event)
                        except Exception as e:
                            logger.error(f"Handler error: {e}")

                self._stats["messages_processed"] += 1

            except Exception as e:
                logger.error(f"Message processing failed: {e}")
                self._stats["messages_failed"] += 1
                await self._send_to_dlq(message, str(e))

    def _get_event_type(
        self,
        topic: str,
        value: Dict[str, Any],
    ) -> Optional[EventType]:
        """Determine event type from topic or value."""
        # Check value first
        if "event_type" in value:
            try:
                return EventType(value["event_type"])
            except ValueError:
                pass

        # Infer from topic
        topic_map = {
            "process-data": EventType.PROCESS_DATA,
            "optimization": EventType.OPTIMIZATION,
            "safety": EventType.SAFETY,
            "efficiency": EventType.EFFICIENCY,
            "emissions": EventType.EMISSIONS,
            "alarms": EventType.ALARM,
        }

        for key, event_type in topic_map.items():
            if key in topic:
                return event_type

        return None

    def _deserialize_event(
        self,
        event_type: EventType,
        value: Dict[str, Any],
    ) -> Optional[BaseEvent]:
        """Deserialize event from dictionary."""
        try:
            # Parse timestamp
            if "timestamp" in value and isinstance(value["timestamp"], str):
                value["timestamp"] = datetime.fromisoformat(
                    value["timestamp"].replace("Z", "+00:00")
                )

            if event_type == EventType.PROCESS_DATA:
                return ProcessDataEvent(**value)
            elif event_type == EventType.OPTIMIZATION:
                return OptimizationEvent(**value)
            elif event_type == EventType.SAFETY:
                return SafetyEvent(**value)
            elif event_type == EventType.EFFICIENCY:
                return EfficiencyEvent(**value)
            elif event_type == EventType.EMISSIONS:
                return EmissionsEvent(**value)
            elif event_type == EventType.ALARM:
                return AlarmEvent(**value)

        except Exception as e:
            logger.error(f"Deserialization error: {e}")

        return None

    async def _send_to_dlq(
        self,
        message: Dict[str, Any],
        reason: str,
    ) -> None:
        """Send failed message to dead letter queue."""
        dlq_entry = {
            "original_message": message,
            "failure_reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if len(self._dlq) >= self._max_dlq_size:
            self._dlq.pop(0)
        self._dlq.append(dlq_entry)

        logger.warning(f"Message sent to DLQ: {reason}")

    async def seek_to_beginning(self, topics: Optional[List[str]] = None) -> None:
        """Seek to beginning of topics."""
        # In production:
        # await self._consumer.seek_to_beginning(*partitions)
        logger.info(f"Seeking to beginning: {topics or 'all'}")

    async def seek_to_end(self, topics: Optional[List[str]] = None) -> None:
        """Seek to end of topics."""
        # In production:
        # await self._consumer.seek_to_end(*partitions)
        logger.info(f"Seeking to end: {topics or 'all'}")

    async def commit(self) -> None:
        """Commit current offsets."""
        # In production:
        # await self._consumer.commit()
        logger.debug("Committed offsets")

    def get_dlq_messages(self, limit: int = 100) -> List[Dict]:
        """Get messages from dead letter queue."""
        return self._dlq[:limit]

    def clear_dlq(self) -> int:
        """Clear dead letter queue."""
        count = len(self._dlq)
        self._dlq.clear()
        return count

    def get_statistics(self) -> Dict:
        """Get consumer statistics."""
        return {
            **self._stats,
            "connected": self._connected,
            "running": self._running,
            "dlq_size": len(self._dlq),
            "active_tasks": len(self._active_tasks),
        }

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_running(self) -> bool:
        return self._running
