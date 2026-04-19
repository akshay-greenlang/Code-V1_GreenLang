"""
Integration Tests: Kafka Streaming

Tests Kafka producer/consumer functionality including:
- Message production and consumption
- Topic management
- Serialization/deserialization
- Error handling and retries
- Consumer group management

Reference: GL-001 Specification Section 11.3
Target Coverage: 85%+
"""

import pytest
import json
import asyncio
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from unittest.mock import AsyncMock, MagicMock, patch
from enum import Enum


# =============================================================================
# Kafka Classes (Simulated Production Code)
# =============================================================================

class KafkaError(Exception):
    """Base Kafka error."""
    pass


class KafkaProducerError(KafkaError):
    """Raised when Kafka production fails."""
    pass


class KafkaConsumerError(KafkaError):
    """Raised when Kafka consumption fails."""
    pass


class SerializationError(KafkaError):
    """Raised when message serialization fails."""
    pass


@dataclass
class KafkaConfig:
    """Kafka client configuration."""
    bootstrap_servers: List[str]
    client_id: str
    security_protocol: str = "SASL_SSL"
    sasl_mechanism: str = "PLAIN"
    sasl_username: Optional[str] = None
    sasl_password: Optional[str] = None
    acks: str = "all"
    retries: int = 3
    retry_backoff_ms: int = 1000
    batch_size: int = 16384
    linger_ms: int = 5


@dataclass
class ThermalDataMessage:
    """Thermal data message for Kafka."""
    message_id: str
    timestamp: str
    boiler_id: str
    temperature: float
    pressure: float
    flow_rate: float
    efficiency: float

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> 'ThermalDataMessage':
        data = json.loads(json_str)
        return cls(**data)


@dataclass
class KafkaMessage:
    """Kafka message wrapper."""
    topic: str
    key: Optional[str]
    value: Any
    headers: Dict[str, str]
    partition: Optional[int] = None
    offset: Optional[int] = None
    timestamp: Optional[datetime] = None


class KafkaProducer:
    """Kafka producer for sending thermal data."""

    def __init__(self, config: KafkaConfig):
        self.config = config
        self.connected = False
        self._send_count = 0
        self._error_count = 0

    async def connect(self) -> bool:
        """Connect to Kafka cluster."""
        self.connected = True
        return True

    async def disconnect(self) -> bool:
        """Disconnect from Kafka cluster."""
        await self.flush()
        self.connected = False
        return True

    async def send(self, topic: str, value: Any, key: Optional[str] = None,
                   headers: Optional[Dict[str, str]] = None,
                   partition: Optional[int] = None) -> Dict[str, Any]:
        """Send message to Kafka topic."""
        if not self.connected:
            raise KafkaProducerError("Producer not connected")

        try:
            # Serialize value if needed
            if isinstance(value, str):
                serialized_value = value.encode('utf-8')
            elif hasattr(value, 'to_json'):
                serialized_value = value.to_json().encode('utf-8')
            else:
                serialized_value = json.dumps(value).encode('utf-8')

            self._send_count += 1

            # Return metadata
            return {
                "topic": topic,
                "partition": partition or 0,
                "offset": self._send_count,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self._error_count += 1
            raise KafkaProducerError(f"Failed to send message: {str(e)}")

    async def send_batch(self, messages: List[KafkaMessage]) -> List[Dict[str, Any]]:
        """Send batch of messages."""
        results = []
        for msg in messages:
            result = await self.send(
                topic=msg.topic,
                value=msg.value,
                key=msg.key,
                headers=msg.headers,
                partition=msg.partition
            )
            results.append(result)
        return results

    async def flush(self, timeout: float = 10.0) -> bool:
        """Flush pending messages."""
        return True

    def get_metrics(self) -> Dict[str, int]:
        """Get producer metrics."""
        return {
            "messages_sent": self._send_count,
            "errors": self._error_count
        }


class KafkaConsumer:
    """Kafka consumer for receiving thermal data."""

    def __init__(self, config: KafkaConfig, group_id: str):
        self.config = config
        self.group_id = group_id
        self.connected = False
        self.subscribed_topics: List[str] = []
        self._message_queue: List[KafkaMessage] = []
        self._offset = 0
        self._committed_offset = 0

    async def connect(self) -> bool:
        """Connect to Kafka cluster."""
        self.connected = True
        return True

    async def disconnect(self) -> bool:
        """Disconnect from Kafka cluster."""
        self.connected = False
        return True

    async def subscribe(self, topics: List[str]) -> bool:
        """Subscribe to topics."""
        if not self.connected:
            raise KafkaConsumerError("Consumer not connected")

        self.subscribed_topics = topics
        return True

    async def unsubscribe(self) -> bool:
        """Unsubscribe from all topics."""
        self.subscribed_topics = []
        return True

    async def poll(self, timeout_ms: int = 1000) -> List[KafkaMessage]:
        """Poll for new messages."""
        if not self.connected:
            raise KafkaConsumerError("Consumer not connected")

        if not self.subscribed_topics:
            raise KafkaConsumerError("No topics subscribed")

        # Return queued messages
        messages = self._message_queue.copy()
        self._message_queue.clear()

        for msg in messages:
            msg.offset = self._offset
            self._offset += 1

        return messages

    async def commit(self, async_commit: bool = True) -> bool:
        """Commit current offset."""
        self._committed_offset = self._offset
        return True

    async def seek(self, topic: str, partition: int, offset: int) -> bool:
        """Seek to specific offset."""
        self._offset = offset
        return True

    def add_test_message(self, message: KafkaMessage):
        """Add message to queue for testing."""
        self._message_queue.append(message)

    def get_position(self) -> int:
        """Get current position."""
        return self._offset

    def get_committed(self) -> int:
        """Get last committed offset."""
        return self._committed_offset


class ThermalDataStreamer:
    """High-level thermal data streaming service."""

    def __init__(self, config: KafkaConfig, topic: str):
        self.config = config
        self.topic = topic
        self.producer = KafkaProducer(config)
        self._running = False

    async def start(self):
        """Start the streamer."""
        await self.producer.connect()
        self._running = True

    async def stop(self):
        """Stop the streamer."""
        self._running = False
        await self.producer.disconnect()

    async def publish_thermal_data(self, data: ThermalDataMessage) -> bool:
        """Publish thermal data message."""
        if not self._running:
            raise KafkaProducerError("Streamer not running")

        await self.producer.send(
            topic=self.topic,
            value=data,
            key=data.boiler_id,
            headers={"content_type": "application/json"}
        )
        return True

    async def publish_batch(self, data_list: List[ThermalDataMessage]) -> int:
        """Publish batch of thermal data messages."""
        count = 0
        for data in data_list:
            await self.publish_thermal_data(data)
            count += 1
        return count


# =============================================================================
# Test Classes
# =============================================================================

@pytest.mark.integration
class TestKafkaProducer:
    """Test Kafka producer functionality."""

    @pytest.fixture
    def config(self):
        """Create Kafka configuration."""
        return KafkaConfig(
            bootstrap_servers=["localhost:9092"],
            client_id="test-producer"
        )

    @pytest.fixture
    async def producer(self, config):
        """Create and connect Kafka producer."""
        producer = KafkaProducer(config)
        await producer.connect()
        yield producer
        await producer.disconnect()

    @pytest.mark.asyncio
    async def test_producer_connect(self, config):
        """Test producer connection."""
        producer = KafkaProducer(config)
        result = await producer.connect()

        assert result == True
        assert producer.connected == True

    @pytest.mark.asyncio
    async def test_producer_disconnect(self, config):
        """Test producer disconnection."""
        producer = KafkaProducer(config)
        await producer.connect()
        result = await producer.disconnect()

        assert result == True
        assert producer.connected == False

    @pytest.mark.asyncio
    async def test_send_string_message(self, producer):
        """Test sending string message."""
        result = await producer.send(
            topic="test-topic",
            value="test message"
        )

        assert result["topic"] == "test-topic"
        assert "offset" in result

    @pytest.mark.asyncio
    async def test_send_json_message(self, producer):
        """Test sending JSON message."""
        data = {"temperature": 450.0, "pressure": 15.0}
        result = await producer.send(
            topic="thermal-data",
            value=data
        )

        assert result["topic"] == "thermal-data"

    @pytest.mark.asyncio
    async def test_send_thermal_data_message(self, producer):
        """Test sending ThermalDataMessage."""
        message = ThermalDataMessage(
            message_id="msg_001",
            timestamp=datetime.now().isoformat(),
            boiler_id="BOILER_001",
            temperature=450.0,
            pressure=15.0,
            flow_rate=500.0,
            efficiency=0.88
        )

        result = await producer.send(
            topic="thermal-data",
            value=message,
            key=message.boiler_id
        )

        assert result["topic"] == "thermal-data"

    @pytest.mark.asyncio
    async def test_send_with_headers(self, producer):
        """Test sending message with headers."""
        result = await producer.send(
            topic="test-topic",
            value="test",
            headers={"content_type": "text/plain", "source": "test"}
        )

        assert result["topic"] == "test-topic"

    @pytest.mark.asyncio
    async def test_send_batch(self, producer):
        """Test sending batch of messages."""
        messages = [
            KafkaMessage(topic="test", key=f"key_{i}", value=f"value_{i}", headers={})
            for i in range(5)
        ]

        results = await producer.send_batch(messages)

        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_send_when_disconnected_fails(self, config):
        """Test sending when not connected raises error."""
        producer = KafkaProducer(config)

        with pytest.raises(KafkaProducerError):
            await producer.send("topic", "value")

    @pytest.mark.asyncio
    async def test_producer_metrics(self, producer):
        """Test producer metrics tracking."""
        await producer.send("topic", "message1")
        await producer.send("topic", "message2")

        metrics = producer.get_metrics()

        assert metrics["messages_sent"] == 2
        assert metrics["errors"] == 0


@pytest.mark.integration
class TestKafkaConsumer:
    """Test Kafka consumer functionality."""

    @pytest.fixture
    def config(self):
        """Create Kafka configuration."""
        return KafkaConfig(
            bootstrap_servers=["localhost:9092"],
            client_id="test-consumer"
        )

    @pytest.fixture
    async def consumer(self, config):
        """Create and connect Kafka consumer."""
        consumer = KafkaConsumer(config, group_id="test-group")
        await consumer.connect()
        yield consumer
        await consumer.disconnect()

    @pytest.mark.asyncio
    async def test_consumer_connect(self, config):
        """Test consumer connection."""
        consumer = KafkaConsumer(config, group_id="test")
        result = await consumer.connect()

        assert result == True
        assert consumer.connected == True

    @pytest.mark.asyncio
    async def test_consumer_subscribe(self, consumer):
        """Test subscribing to topics."""
        result = await consumer.subscribe(["topic1", "topic2"])

        assert result == True
        assert "topic1" in consumer.subscribed_topics
        assert "topic2" in consumer.subscribed_topics

    @pytest.mark.asyncio
    async def test_consumer_unsubscribe(self, consumer):
        """Test unsubscribing from topics."""
        await consumer.subscribe(["topic1"])
        result = await consumer.unsubscribe()

        assert result == True
        assert len(consumer.subscribed_topics) == 0

    @pytest.mark.asyncio
    async def test_consumer_poll(self, consumer):
        """Test polling for messages."""
        await consumer.subscribe(["test-topic"])

        # Add test message
        consumer.add_test_message(KafkaMessage(
            topic="test-topic",
            key="key1",
            value="test value",
            headers={}
        ))

        messages = await consumer.poll()

        assert len(messages) == 1
        assert messages[0].value == "test value"

    @pytest.mark.asyncio
    async def test_consumer_poll_empty(self, consumer):
        """Test polling when no messages available."""
        await consumer.subscribe(["test-topic"])

        messages = await consumer.poll()

        assert len(messages) == 0

    @pytest.mark.asyncio
    async def test_consumer_poll_without_subscription_fails(self, consumer):
        """Test polling without subscription raises error."""
        with pytest.raises(KafkaConsumerError):
            await consumer.poll()

    @pytest.mark.asyncio
    async def test_consumer_commit(self, consumer):
        """Test committing offsets."""
        await consumer.subscribe(["test"])
        consumer.add_test_message(KafkaMessage("test", "k", "v", {}))
        await consumer.poll()

        result = await consumer.commit()

        assert result == True
        assert consumer.get_committed() == consumer.get_position()

    @pytest.mark.asyncio
    async def test_consumer_seek(self, consumer):
        """Test seeking to offset."""
        await consumer.subscribe(["test"])

        result = await consumer.seek("test", 0, 100)

        assert result == True
        assert consumer.get_position() == 100


@pytest.mark.integration
class TestThermalDataStreamer:
    """Test thermal data streaming service."""

    @pytest.fixture
    def config(self):
        """Create Kafka configuration."""
        return KafkaConfig(
            bootstrap_servers=["localhost:9092"],
            client_id="thermal-streamer"
        )

    @pytest.fixture
    async def streamer(self, config):
        """Create and start thermal data streamer."""
        streamer = ThermalDataStreamer(config, topic="thermal-data")
        await streamer.start()
        yield streamer
        await streamer.stop()

    @pytest.mark.asyncio
    async def test_streamer_start(self, config):
        """Test starting streamer."""
        streamer = ThermalDataStreamer(config, topic="test")
        await streamer.start()

        assert streamer._running == True
        await streamer.stop()

    @pytest.mark.asyncio
    async def test_streamer_stop(self, config):
        """Test stopping streamer."""
        streamer = ThermalDataStreamer(config, topic="test")
        await streamer.start()
        await streamer.stop()

        assert streamer._running == False

    @pytest.mark.asyncio
    async def test_publish_thermal_data(self, streamer):
        """Test publishing thermal data."""
        data = ThermalDataMessage(
            message_id="msg_001",
            timestamp=datetime.now().isoformat(),
            boiler_id="BOILER_001",
            temperature=450.0,
            pressure=15.0,
            flow_rate=500.0,
            efficiency=0.88
        )

        result = await streamer.publish_thermal_data(data)

        assert result == True

    @pytest.mark.asyncio
    async def test_publish_batch(self, streamer):
        """Test publishing batch of thermal data."""
        data_list = [
            ThermalDataMessage(
                message_id=f"msg_{i}",
                timestamp=datetime.now().isoformat(),
                boiler_id=f"BOILER_{i:03d}",
                temperature=450.0 + i,
                pressure=15.0,
                flow_rate=500.0,
                efficiency=0.88
            )
            for i in range(10)
        ]

        count = await streamer.publish_batch(data_list)

        assert count == 10

    @pytest.mark.asyncio
    async def test_publish_when_not_running_fails(self, config):
        """Test publishing when streamer not running raises error."""
        streamer = ThermalDataStreamer(config, topic="test")
        data = ThermalDataMessage(
            message_id="msg",
            timestamp=datetime.now().isoformat(),
            boiler_id="BOILER_001",
            temperature=450.0,
            pressure=15.0,
            flow_rate=500.0,
            efficiency=0.88
        )

        with pytest.raises(KafkaProducerError):
            await streamer.publish_thermal_data(data)


@pytest.mark.integration
class TestMessageSerialization:
    """Test message serialization/deserialization."""

    def test_thermal_message_to_json(self):
        """Test ThermalDataMessage serialization to JSON."""
        message = ThermalDataMessage(
            message_id="msg_001",
            timestamp="2025-01-15T10:30:00",
            boiler_id="BOILER_001",
            temperature=450.0,
            pressure=15.0,
            flow_rate=500.0,
            efficiency=0.88
        )

        json_str = message.to_json()
        parsed = json.loads(json_str)

        assert parsed["message_id"] == "msg_001"
        assert parsed["temperature"] == 450.0

    def test_thermal_message_from_json(self):
        """Test ThermalDataMessage deserialization from JSON."""
        json_str = json.dumps({
            "message_id": "msg_002",
            "timestamp": "2025-01-15T10:30:00",
            "boiler_id": "BOILER_002",
            "temperature": 455.0,
            "pressure": 16.0,
            "flow_rate": 510.0,
            "efficiency": 0.87
        })

        message = ThermalDataMessage.from_json(json_str)

        assert message.message_id == "msg_002"
        assert message.temperature == 455.0
        assert message.boiler_id == "BOILER_002"

    def test_json_round_trip(self):
        """Test JSON serialization round trip."""
        original = ThermalDataMessage(
            message_id="msg_003",
            timestamp="2025-01-15T10:30:00",
            boiler_id="BOILER_003",
            temperature=460.0,
            pressure=17.0,
            flow_rate=520.0,
            efficiency=0.86
        )

        json_str = original.to_json()
        restored = ThermalDataMessage.from_json(json_str)

        assert restored.message_id == original.message_id
        assert restored.temperature == original.temperature
        assert restored.boiler_id == original.boiler_id
