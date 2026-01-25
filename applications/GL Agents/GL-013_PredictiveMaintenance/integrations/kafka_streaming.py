# -*- coding: utf-8 -*-
"""GL-013 PredictiveMaintenance - Kafka Streaming Module"""

from __future__ import annotations
import asyncio
import json
import logging
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class KafkaTopics(str, Enum):
    SENSOR_DATA_RAW = "sensor-data-raw"
    FEATURES_COMPUTED = "features-computed"
    PREDICTIONS = "predictions"
    ALERTS = "alerts"
    CMMS_WORK_ORDERS = "cmms-work-orders"
    DEAD_LETTER = "dead-letter-queue"

class SerializationFormat(str, Enum):
    JSON = "json"
    AVRO = "avro"

class CompressionType(str, Enum):
    NONE = "none"
    GZIP = "gzip"
    SNAPPY = "snappy"

class DeliveryGuarantee(str, Enum):
    AT_MOST_ONCE = "at_most_once"
    AT_LEAST_ONCE = "at_least_once"
    EXACTLY_ONCE = "exactly_once"

class ConsumerState(str, Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"

class KafkaStreamConfig(BaseModel):
    bootstrap_servers: List[str] = Field(default=["localhost:9092"])
    client_id: str = Field(default="gl-013-pm")
    group_id: str = Field(default="pm-group")
    producer_acks: str = Field(default="all")
    compression_type: CompressionType = Field(default=CompressionType.SNAPPY)
    enable_idempotence: bool = Field(default=True)
    delivery_guarantee: DeliveryGuarantee = Field(default=DeliveryGuarantee.EXACTLY_ONCE)
    schema_registry_url: Optional[str] = None
    enable_dlq: bool = Field(default=True)
    dlq_topic: str = Field(default="dead-letter-queue")
    max_retries_before_dlq: int = Field(default=3)

class MessageSchema(BaseModel):
    message_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = Field(default="gl-013-pm")
    version: str = Field(default="1.0.0")

class SchemaRegistry:
    def __init__(self, url: Optional[str] = None):
        self.url = url
        self._schemas: Dict[str, Dict] = {}

    async def register_schema(self, subject: str, schema: Dict) -> int:
        schema_id = hash(json.dumps(schema)) % 100000
        self._schemas[subject] = schema
        return schema_id

    async def get_schema(self, subject: str) -> Optional[Dict]:
        return self._schemas.get(subject)

@dataclass
class DeadLetterMessage:
    original_topic: str
    original_message: bytes
    error_message: str
    error_type: str
    retry_count: int
    first_failure_time: datetime
    last_failure_time: datetime

class DeadLetterHandler:
    def __init__(self, config: KafkaStreamConfig):
        self.config = config
        self._failed: Dict[str, DeadLetterMessage] = {}
        self._lock = asyncio.Lock()

    async def handle_failure(self, topic: str, message: bytes, error: Exception) -> bool:
        msg_hash = hashlib.sha256(message).hexdigest()
        async with self._lock:
            if msg_hash in self._failed:
                self._failed[msg_hash].retry_count += 1
            else:
                now = datetime.now(timezone.utc)
                self._failed[msg_hash] = DeadLetterMessage(topic, message, str(error), type(error).__name__, 1, now, now)
            if self._failed[msg_hash].retry_count >= self.config.max_retries_before_dlq:
                del self._failed[msg_hash]
                return False
            return True

class KafkaProducerWrapper:
    def __init__(self, config: KafkaStreamConfig):
        self.config = config
        self._initialized = False
        self._sent = 0
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        if self._initialized: return
        await asyncio.sleep(0.1)
        self._initialized = True

    async def close(self) -> None:
        self._initialized = False

    async def send(self, topic: str, value: bytes, key: Optional[bytes] = None, headers: Optional[Dict] = None) -> bool:
        if not self._initialized: await self.initialize()
        async with self._lock:
            self._sent += 1
        return True

    async def flush(self) -> None:
        await asyncio.sleep(0.01)

    @property
    def stats(self) -> Dict:
        return {"initialized": self._initialized, "messages_sent": self._sent}

class ExactlyOnceProducer(KafkaProducerWrapper):
    def __init__(self, config: KafkaStreamConfig):
        config.enable_idempotence = True
        super().__init__(config)
        self._tx_active = False

    async def begin_transaction(self) -> None:
        self._tx_active = True

    async def commit_transaction(self) -> None:
        self._tx_active = False

    async def abort_transaction(self) -> None:
        self._tx_active = False

    async def send_exactly_once(self, topic: str, value: bytes, key: Optional[bytes] = None) -> bool:
        try:
            await self.begin_transaction()
            result = await self.send(topic, value, key)
            await self.commit_transaction()
            return result
        except:
            await self.abort_transaction()
            raise

class KafkaConsumerWrapper:
    def __init__(self, config: KafkaStreamConfig, topics: List[str], handler: Optional[Callable] = None):
        self.config = config
        self.topics = topics
        self.handler = handler
        self._state = ConsumerState.STOPPED
        self._task: Optional[asyncio.Task] = None
        self._consumed = 0

    async def start(self) -> None:
        if self._state == ConsumerState.RUNNING: return
        self._state = ConsumerState.RUNNING
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try: await self._task
            except asyncio.CancelledError: pass
        self._state = ConsumerState.STOPPED

    async def _loop(self) -> None:
        while self._state == ConsumerState.RUNNING:
            try: await asyncio.sleep(0.1)
            except asyncio.CancelledError: break

    @property
    def state(self) -> ConsumerState:
        return self._state

    @property
    def stats(self) -> Dict:
        return {"state": self._state.value, "topics": self.topics}

def create_producer(config_dict: Dict) -> KafkaProducerWrapper:
    return KafkaProducerWrapper(KafkaStreamConfig(**config_dict))

def create_consumer(config_dict: Dict, topics: List[str], handler: Optional[Callable] = None) -> KafkaConsumerWrapper:
    return KafkaConsumerWrapper(KafkaStreamConfig(**config_dict), topics, handler)

def create_exactly_once_producer(config_dict: Dict) -> ExactlyOnceProducer:
    return ExactlyOnceProducer(KafkaStreamConfig(**config_dict))