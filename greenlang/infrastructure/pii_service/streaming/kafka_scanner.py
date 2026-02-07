# -*- coding: utf-8 -*-
"""
Kafka PII Scanner - SEC-011 Streaming PII Detection

Real-time PII scanning for Apache Kafka streams using aiokafka.
Consumes messages from configured input topics, scans for PII,
and routes to output or dead letter queue based on enforcement policy.

Architecture:
    KafkaPIIScanner
    ├── AIOKafkaConsumer (input topics)
    ├── AIOKafkaProducer (output + DLQ)
    ├── BaseStreamProcessor (processing logic)
    └── PIIEnforcementEngine (detection + policy)

Message Flow:
    1. Consumer polls messages from input topics
    2. For each message, extract content and metadata
    3. Call process_message() (inherited from BaseStreamProcessor)
    4. Based on result:
       - allowed: forward to output topic unchanged
       - redacted: forward modified content to output topic
       - blocked: send to DLQ with detection details
    5. Commit offset on success

Features:
    - Async/await with aiokafka for high throughput
    - SASL/SSL authentication support
    - Dead letter queue for blocked messages
    - Preserves message headers and keys
    - Graceful shutdown with timeout
    - Prometheus metrics integration
    - Multi-tenant support via headers

Example:
    >>> from greenlang.infrastructure.pii_service.streaming import (
    ...     KafkaPIIScanner,
    ...     StreamingConfig,
    ... )
    >>> config = StreamingConfig()
    >>> scanner = KafkaPIIScanner(enforcement_engine, config)
    >>> await scanner.start()  # Runs until stop() is called

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-011 PII Detection/Redaction Enhancements
Status: Production Ready
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from greenlang.infrastructure.pii_service.streaming.config import (
    KafkaConfig,
    StreamingConfig,
)
from greenlang.infrastructure.pii_service.streaming.metrics import (
    StreamingPIIMetrics,
    get_streaming_metrics,
)
from greenlang.infrastructure.pii_service.streaming.stream_processor import (
    BaseStreamProcessor,
    EnforcementContext,
    PIIEnforcementEngine,
    ProcessingResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type hints for aiokafka (avoid hard dependency at import time)
# ---------------------------------------------------------------------------

try:
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
    from aiokafka.structs import ConsumerRecord

    AIOKAFKA_AVAILABLE = True
except ImportError:
    AIOKafkaConsumer = None  # type: ignore
    AIOKafkaProducer = None  # type: ignore
    ConsumerRecord = None  # type: ignore
    AIOKAFKA_AVAILABLE = False
    logger.warning("aiokafka not installed - Kafka scanner will not be available")


# ---------------------------------------------------------------------------
# Kafka PII Scanner
# ---------------------------------------------------------------------------


class KafkaPIIScanner(BaseStreamProcessor):
    """Real-time PII scanning for Kafka streams.

    Consumes messages from configured Kafka topics, scans for PII using
    the enforcement engine, and routes messages based on the result:
    - Clean messages go to the output topic
    - Blocked messages go to the dead letter queue

    This scanner inherits processing logic from BaseStreamProcessor and
    implements Kafka-specific consumer/producer management.

    Attributes:
        kafka_config: Kafka-specific configuration.

    Example:
        >>> # Basic usage
        >>> config = StreamingConfig(
        ...     backend="kafka",
        ...     kafka=KafkaConfig(
        ...         bootstrap_servers=["kafka:9092"],
        ...         input_topics=["events.raw"],
        ...         output_topic="events.clean",
        ...     ),
        ... )
        >>> scanner = KafkaPIIScanner(enforcement_engine, config)
        >>> await scanner.start()

        >>> # With custom configuration
        >>> kafka_config = KafkaConfig(
        ...     bootstrap_servers=["kafka1:9092", "kafka2:9092"],
        ...     consumer_group="pii-scanner-prod",
        ...     security_protocol="SASL_SSL",
        ...     sasl_mechanism="SCRAM-SHA-256",
        ...     sasl_username="scanner",
        ...     sasl_password="secret",
        ... )
        >>> config = StreamingConfig(kafka=kafka_config)
        >>> scanner = KafkaPIIScanner(enforcement_engine, config)
    """

    def __init__(
        self,
        enforcement_engine: PIIEnforcementEngine,
        config: StreamingConfig,
        metrics: Optional[StreamingPIIMetrics] = None,
    ):
        """Initialize the Kafka PII scanner.

        Args:
            enforcement_engine: PII enforcement engine for scanning.
            config: Streaming configuration with Kafka settings.
            metrics: Optional metrics instance.

        Raises:
            ImportError: If aiokafka is not installed.
        """
        if not AIOKAFKA_AVAILABLE:
            raise ImportError(
                "aiokafka is required for Kafka streaming. "
                "Install with: pip install aiokafka"
            )

        super().__init__(
            enforcement_engine=enforcement_engine,
            config=config,
            metrics=metrics or get_streaming_metrics(backend="kafka"),
        )

        self._kafka_config = config.kafka
        self._consumer: Optional[AIOKafkaConsumer] = None
        self._producer: Optional[AIOKafkaProducer] = None
        self._consume_task: Optional[asyncio.Task] = None

        logger.info(
            "Initialized Kafka PII Scanner: topics=%s group=%s",
            self._kafka_config.input_topics,
            self._kafka_config.consumer_group,
        )

    async def start(self) -> None:
        """Start the Kafka consumer and producer.

        Initializes connections to Kafka brokers, subscribes to input topics,
        and begins processing messages in a background task.

        Raises:
            ConnectionError: If unable to connect to Kafka brokers.
            RuntimeError: If scanner is already running.
        """
        if self._running:
            raise RuntimeError("Kafka scanner is already running")

        logger.info(
            "Starting Kafka PII Scanner on topics: %s",
            self._kafka_config.input_topics,
        )

        try:
            # Create consumer
            self._consumer = AIOKafkaConsumer(
                *self._kafka_config.input_topics,
                bootstrap_servers=self._kafka_config.bootstrap_servers,
                group_id=self._kafka_config.consumer_group,
                auto_offset_reset=self._kafka_config.auto_offset_reset,
                enable_auto_commit=self._kafka_config.enable_auto_commit,
                session_timeout_ms=self._kafka_config.session_timeout_ms,
                max_poll_records=self._kafka_config.max_poll_records,
            )

            # Create producer
            self._producer = AIOKafkaProducer(
                bootstrap_servers=self._kafka_config.bootstrap_servers,
                acks=self._kafka_config.producer_acks,
            )

            # Start both
            await self._consumer.start()
            await self._producer.start()

            self._running = True
            self._start_time = time.monotonic()

            # Update metrics
            self._metrics.set_running(self._kafka_config.consumer_group, True)

            logger.info(
                "Kafka PII Scanner started successfully: consumer_group=%s",
                self._kafka_config.consumer_group,
            )

            # Start consuming
            await self._consume_loop()

        except Exception as e:
            logger.error("Failed to start Kafka scanner: %s", str(e), exc_info=True)
            await self._cleanup()
            raise ConnectionError(f"Failed to connect to Kafka: {str(e)}") from e

    async def start_background(self) -> None:
        """Start the scanner in a background task.

        Returns immediately after starting the consumer. Use this when
        you need to run the scanner alongside other async tasks.

        Example:
            >>> await scanner.start_background()
            >>> # Do other work while scanner runs
            >>> await scanner.stop()
        """
        if self._running:
            raise RuntimeError("Kafka scanner is already running")

        self._consume_task = asyncio.create_task(self.start())

    async def stop(self) -> None:
        """Stop the Kafka consumer and producer.

        Gracefully shuts down the scanner, waiting for in-flight
        messages to complete before closing connections.
        """
        if not self._running:
            logger.warning("Kafka scanner is not running")
            return

        logger.info("Stopping Kafka PII Scanner...")
        self._running = False

        # Cancel background task if exists
        if self._consume_task and not self._consume_task.done():
            self._consume_task.cancel()
            try:
                await asyncio.wait_for(
                    self._consume_task,
                    timeout=self._config.shutdown_timeout_seconds,
                )
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        await self._cleanup()

        # Update metrics
        self._metrics.set_running(self._kafka_config.consumer_group, False)

        logger.info(
            "Kafka PII Scanner stopped: processed=%d blocked=%d errors=%d",
            self._processed_count,
            self._blocked_count,
            self._error_count,
        )

    async def _cleanup(self) -> None:
        """Clean up consumer and producer resources."""
        if self._consumer:
            try:
                await self._consumer.stop()
            except Exception as e:
                logger.warning("Error stopping consumer: %s", str(e))
            self._consumer = None

        if self._producer:
            try:
                await self._producer.stop()
            except Exception as e:
                logger.warning("Error stopping producer: %s", str(e))
            self._producer = None

    async def _consume_loop(self) -> None:
        """Main consume loop that processes messages.

        Continuously polls for messages and processes them until
        stop() is called.
        """
        try:
            async for message in self._consumer:
                if not self._running:
                    break

                await self._process_kafka_message(message)

        except asyncio.CancelledError:
            logger.info("Consume loop cancelled")
        except Exception as e:
            logger.error("Error in consume loop: %s", str(e), exc_info=True)
            self._error_count += 1
        finally:
            await self._cleanup()

    async def _process_kafka_message(self, message: ConsumerRecord) -> None:
        """Process a single Kafka message.

        Extracts content and metadata from the Kafka message,
        processes it through the enforcement engine, and routes
        the result to the appropriate output topic.

        Args:
            message: Kafka consumer record to process.
        """
        try:
            # Decode message content
            content = message.value.decode("utf-8")

            # Build metadata from message
            metadata = self._extract_metadata(message)

            # Record message size
            self._metrics.record_message_size(
                message.topic, len(message.value)
            )

            # Process through enforcement engine (inherited method)
            result = await self.process_message(content, metadata)

            # Route based on result
            if result.action == "blocked":
                await self._send_to_dlq(message, result)
            elif result.action in ("allowed", "redacted"):
                await self._send_to_output(message, result)
            # Error case: don't forward, just log (already logged in process_message)

            # Update consumer offset tracking
            self._metrics.set_consumer_offset(
                message.topic, message.partition, message.offset
            )

        except UnicodeDecodeError as e:
            self._error_count += 1
            self._metrics.record_error(message.topic, "decode_error")
            logger.error(
                "Failed to decode message at %s:%d:%d: %s",
                message.topic,
                message.partition,
                message.offset,
                str(e),
            )

        except Exception as e:
            self._error_count += 1
            self._metrics.record_error(message.topic, type(e).__name__)
            logger.error(
                "Error processing Kafka message at %s:%d:%d: %s",
                message.topic,
                message.partition,
                message.offset,
                str(e),
                exc_info=True,
            )

    def _extract_metadata(self, message: ConsumerRecord) -> Dict[str, Any]:
        """Extract metadata from a Kafka message.

        Args:
            message: Kafka consumer record.

        Returns:
            Dictionary of message metadata.
        """
        metadata: Dict[str, Any] = {
            "message_id": f"{message.topic}:{message.partition}:{message.offset}",
            "topic": message.topic,
            "partition": message.partition,
            "offset": message.offset,
            "timestamp": message.timestamp,
            "timestamp_type": message.timestamp_type,
            "key": message.key.decode("utf-8") if message.key else None,
        }

        # Extract tenant_id and other metadata from headers
        if message.headers:
            for key, value in message.headers:
                if key == "tenant_id":
                    metadata["tenant_id"] = value.decode("utf-8")
                elif key == "source":
                    metadata["source"] = value.decode("utf-8")
                elif key == "correlation_id":
                    metadata["correlation_id"] = value.decode("utf-8")
                elif key == "trace_id":
                    metadata["trace_id"] = value.decode("utf-8")

        return metadata

    async def _send_to_output(
        self,
        original_message: ConsumerRecord,
        result: ProcessingResult,
    ) -> None:
        """Send processed message to output topic.

        Args:
            original_message: Original Kafka message.
            result: Processing result with potentially modified content.
        """
        # Determine content to send
        content = result.modified_content
        if content is None:
            # If no modification, use original
            content = original_message.value.decode("utf-8")

        # Preserve original headers and add scan metadata
        headers = list(original_message.headers) if original_message.headers else []

        if self._config.add_scan_metadata:
            headers.extend([
                ("pii_scanned", b"true"),
                ("pii_action", result.action.encode("utf-8")),
                ("pii_scan_time_ms", str(result.processing_time_ms).encode("utf-8")),
            ])

            if result.detections:
                detection_types = ",".join(d.pii_type for d in result.detections)
                headers.append(("pii_types_found", detection_types.encode("utf-8")))

        await self._producer.send(
            self._kafka_config.output_topic,
            key=original_message.key,
            value=content.encode("utf-8"),
            headers=headers,
        )

    async def _send_to_dlq(
        self,
        original_message: ConsumerRecord,
        result: ProcessingResult,
    ) -> None:
        """Send blocked message to dead letter queue.

        Creates a DLQ record with the original message, detection details,
        and metadata for investigation.

        Args:
            original_message: Original Kafka message.
            result: Processing result with block reason.
        """
        dlq_record = {
            "original_message": original_message.value.decode("utf-8"),
            "detections": [
                {
                    "pii_type": d.pii_type,
                    "confidence": d.confidence,
                    "start": d.start,
                    "end": d.end,
                }
                for d in result.detections
            ],
            "reason": "PII_BLOCKED",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": {
                "topic": original_message.topic,
                "partition": original_message.partition,
                "offset": original_message.offset,
                "key": original_message.key.decode("utf-8") if original_message.key else None,
            },
            "processing_time_ms": result.processing_time_ms,
        }

        # Add headers from original message
        if original_message.headers:
            dlq_record["original_headers"] = {
                k: v.decode("utf-8") if isinstance(v, bytes) else v
                for k, v in original_message.headers
            }

        await self._producer.send(
            self._kafka_config.dlq_topic,
            key=original_message.key,
            value=json.dumps(dlq_record).encode("utf-8"),
            headers=[
                ("original_topic", original_message.topic.encode("utf-8")),
                ("block_reason", b"pii_detected"),
            ],
        )

        # Record DLQ metric
        self._metrics.record_dlq_message(original_message.topic, "pii_blocked")

        logger.warning(
            "Message blocked and sent to DLQ: %s:%d:%d - %d PII detections",
            original_message.topic,
            original_message.partition,
            original_message.offset,
            len(result.detections),
        )


# ---------------------------------------------------------------------------
# Factory Function
# ---------------------------------------------------------------------------


def create_kafka_scanner(
    enforcement_engine: PIIEnforcementEngine,
    config: Optional[StreamingConfig] = None,
    **kafka_overrides: Any,
) -> KafkaPIIScanner:
    """Factory function to create a Kafka PII scanner.

    Convenience function that creates a scanner with sensible defaults
    and allows for easy configuration overrides.

    Args:
        enforcement_engine: PII enforcement engine instance.
        config: Optional streaming configuration.
        **kafka_overrides: Override specific Kafka settings.

    Returns:
        Configured KafkaPIIScanner instance.

    Example:
        >>> scanner = create_kafka_scanner(
        ...     enforcement_engine,
        ...     bootstrap_servers=["kafka:9092"],
        ...     input_topics=["events.raw"],
        ...     consumer_group="my-scanner",
        ... )
    """
    if config is None:
        config = StreamingConfig()

    # Apply overrides to Kafka config
    if kafka_overrides:
        kafka_dict = config.kafka.model_dump()
        kafka_dict.update(kafka_overrides)
        config = StreamingConfig(
            **{**config.model_dump(), "kafka": KafkaConfig(**kafka_dict)}
        )

    return KafkaPIIScanner(enforcement_engine, config)


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "KafkaPIIScanner",
    "create_kafka_scanner",
    "AIOKAFKA_AVAILABLE",
]
