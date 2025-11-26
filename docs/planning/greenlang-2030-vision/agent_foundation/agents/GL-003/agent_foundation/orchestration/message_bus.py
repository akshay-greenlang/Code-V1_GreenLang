# -*- coding: utf-8 -*-
"""
Message bus for multi-agent communication.

Provides publish-subscribe messaging capabilities for coordinating
multiple GreenLang agents in a distributed system.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
from enum import Enum
import asyncio
import logging
import uuid

logger = logging.getLogger(__name__)


class MessagePriority(Enum):
    """Message priority levels."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


@dataclass
class Message:
    """
    Message for inter-agent communication.

    Attributes:
        sender_id: ID of the sending agent
        recipient_id: ID of the receiving agent (or '*' for broadcast)
        message_type: Type of message (command, event, query, response)
        payload: Message payload data
        priority: Message priority level
        message_id: Unique message identifier
        timestamp: Message creation timestamp
        correlation_id: ID for correlating request/response pairs
    """
    sender_id: str
    recipient_id: str
    message_type: str
    payload: Dict[str, Any]
    priority: int = MessagePriority.NORMAL.value
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            'sender_id': self.sender_id,
            'recipient_id': self.recipient_id,
            'message_type': self.message_type,
            'payload': self.payload,
            'priority': self.priority,
            'message_id': self.message_id,
            'timestamp': self.timestamp.isoformat(),
            'correlation_id': self.correlation_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary."""
        return cls(
            sender_id=data['sender_id'],
            recipient_id=data['recipient_id'],
            message_type=data['message_type'],
            payload=data['payload'],
            priority=data.get('priority', MessagePriority.NORMAL.value),
            message_id=data.get('message_id', str(uuid.uuid4())),
            timestamp=datetime.fromisoformat(data['timestamp']) if 'timestamp' in data else datetime.now(timezone.utc),
            correlation_id=data.get('correlation_id')
        )


class MessageBus:
    """
    Asynchronous message bus for agent communication.

    Provides publish-subscribe messaging with topic-based routing,
    priority queuing, and message persistence.

    Example:
        >>> bus = MessageBus()
        >>> await bus.subscribe("agent.GL-003", callback)
        >>> await bus.publish("agent.GL-003", message)
    """

    def __init__(self, max_queue_size: int = 10000):
        """
        Initialize message bus.

        Args:
            max_queue_size: Maximum messages in queue
        """
        self._subscriptions: Dict[str, List[Callable]] = {}
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_queue_size)
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None
        self._message_history: List[Message] = []
        self._max_history = 1000

        logger.info("MessageBus initialized")

    async def start(self) -> None:
        """Start the message bus processor."""
        if not self._running:
            self._running = True
            self._processor_task = asyncio.create_task(self._process_messages())
            logger.info("MessageBus started")

    async def stop(self) -> None:
        """Stop the message bus processor."""
        self._running = False
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        logger.info("MessageBus stopped")

    async def close(self) -> None:
        """Close the message bus and release resources."""
        await self.stop()
        self._subscriptions.clear()
        self._message_history.clear()
        logger.info("MessageBus closed")

    async def subscribe(self, topic: str, callback: Callable[[Message], Any]) -> None:
        """
        Subscribe to a topic.

        Args:
            topic: Topic pattern to subscribe to
            callback: Async callback function for messages
        """
        if topic not in self._subscriptions:
            self._subscriptions[topic] = []
        self._subscriptions[topic].append(callback)
        logger.debug(f"Subscribed to topic: {topic}")

    async def unsubscribe(self, topic: str, callback: Callable[[Message], Any]) -> None:
        """
        Unsubscribe from a topic.

        Args:
            topic: Topic to unsubscribe from
            callback: Callback to remove
        """
        if topic in self._subscriptions:
            self._subscriptions[topic] = [
                cb for cb in self._subscriptions[topic] if cb != callback
            ]
            if not self._subscriptions[topic]:
                del self._subscriptions[topic]
            logger.debug(f"Unsubscribed from topic: {topic}")

    async def publish(self, topic: str, message: Message) -> None:
        """
        Publish a message to a topic.

        Args:
            topic: Topic to publish to
            message: Message to publish
        """
        # Add to queue with priority (lower number = higher priority)
        await self._queue.put((message.priority, message.timestamp.timestamp(), topic, message))

        # Store in history
        self._message_history.append(message)
        if len(self._message_history) > self._max_history:
            self._message_history.pop(0)

        logger.debug(f"Published message {message.message_id} to topic: {topic}")

    async def _process_messages(self) -> None:
        """Process messages from the queue."""
        while self._running:
            try:
                # Wait for message with timeout
                try:
                    priority, timestamp, topic, message = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Find matching subscribers
                for sub_topic, callbacks in self._subscriptions.items():
                    if self._topic_matches(topic, sub_topic):
                        for callback in callbacks:
                            try:
                                if asyncio.iscoroutinefunction(callback):
                                    await callback(message)
                                else:
                                    callback(message)
                            except Exception as e:
                                logger.error(f"Error in message callback: {e}")

                self._queue.task_done()

            except Exception as e:
                logger.error(f"Error processing message: {e}")

    def _topic_matches(self, topic: str, pattern: str) -> bool:
        """Check if topic matches subscription pattern."""
        if pattern == '*':
            return True
        if pattern.endswith('.*'):
            prefix = pattern[:-2]
            return topic.startswith(prefix)
        return topic == pattern

    def get_stats(self) -> Dict[str, Any]:
        """Get message bus statistics."""
        return {
            'running': self._running,
            'queue_size': self._queue.qsize(),
            'subscription_count': len(self._subscriptions),
            'topics': list(self._subscriptions.keys()),
            'history_size': len(self._message_history)
        }
