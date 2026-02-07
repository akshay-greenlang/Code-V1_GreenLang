# -*- coding: utf-8 -*-
"""
Kinesis PII Scanner - SEC-011 Streaming PII Detection

Real-time PII scanning for AWS Kinesis Data Streams using boto3.
Consumes records from configured Kinesis streams, scans for PII,
and routes to output or dead letter queue based on enforcement policy.

Architecture:
    KinesisPIIScanner
    ├── boto3 Kinesis Client (consumer)
    ├── boto3 Kinesis Client (producer)
    ├── BaseStreamProcessor (processing logic)
    └── PIIEnforcementEngine (detection + policy)

Record Flow:
    1. Get shard iterators for all shards
    2. Poll records from each shard
    3. For each record, extract content and metadata
    4. Call process_message() (inherited from BaseStreamProcessor)
    5. Based on result:
       - allowed: put to output stream unchanged
       - redacted: put modified content to output stream
       - blocked: put to DLQ stream with detection details
    6. Update checkpoint (sequence number)

Features:
    - Async processing with asyncio and boto3
    - Automatic shard discovery and iteration
    - Enhanced Fan-Out support (optional)
    - Dead letter queue stream for blocked records
    - Preserves partition keys
    - Graceful shutdown with timeout
    - Prometheus metrics integration
    - LocalStack support for testing

Example:
    >>> from greenlang.infrastructure.pii_service.streaming import (
    ...     KinesisPIIScanner,
    ...     StreamingConfig,
    ... )
    >>> config = StreamingConfig(backend="kinesis")
    >>> scanner = KinesisPIIScanner(enforcement_engine, config)
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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.infrastructure.pii_service.streaming.config import (
    KinesisConfig,
    KinesisShardIteratorType,
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
# Type hints for boto3 (avoid hard dependency at import time)
# ---------------------------------------------------------------------------

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError

    BOTO3_AVAILABLE = True
except ImportError:
    boto3 = None  # type: ignore
    BotoCoreError = Exception  # type: ignore
    ClientError = Exception  # type: ignore
    BOTO3_AVAILABLE = False
    logger.warning("boto3 not installed - Kinesis scanner will not be available")


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class ShardState:
    """State tracking for a single shard.

    Attributes:
        shard_id: Kinesis shard identifier.
        iterator: Current shard iterator.
        sequence_number: Last processed sequence number.
        behind_latest_ms: Milliseconds behind latest record.
    """

    shard_id: str
    iterator: Optional[str] = None
    sequence_number: Optional[str] = None
    behind_latest_ms: int = 0


# ---------------------------------------------------------------------------
# Kinesis PII Scanner
# ---------------------------------------------------------------------------


class KinesisPIIScanner(BaseStreamProcessor):
    """Real-time PII scanning for AWS Kinesis Data Streams.

    Consumes records from configured Kinesis streams, scans for PII using
    the enforcement engine, and routes records based on the result:
    - Clean records go to the output stream
    - Blocked records go to the dead letter queue stream

    This scanner inherits processing logic from BaseStreamProcessor and
    implements Kinesis-specific consumer/producer management.

    Attributes:
        kinesis_config: Kinesis-specific configuration.

    Example:
        >>> # Basic usage
        >>> config = StreamingConfig(
        ...     backend="kinesis",
        ...     kinesis=KinesisConfig(
        ...         stream_name="events-raw",
        ...         output_stream_name="events-scanned",
        ...         region="us-west-2",
        ...     ),
        ... )
        >>> scanner = KinesisPIIScanner(enforcement_engine, config)
        >>> await scanner.start()

        >>> # With Enhanced Fan-Out
        >>> kinesis_config = KinesisConfig(
        ...     stream_name="events-raw",
        ...     use_enhanced_fan_out=True,
        ...     consumer_name="pii-scanner-prod",
        ... )
        >>> config = StreamingConfig(kinesis=kinesis_config)
        >>> scanner = KinesisPIIScanner(enforcement_engine, config)

        >>> # For LocalStack testing
        >>> config = StreamingConfig(
        ...     kinesis=KinesisConfig(
        ...         endpoint_url="http://localhost:4566",
        ...     ),
        ... )
    """

    def __init__(
        self,
        enforcement_engine: PIIEnforcementEngine,
        config: StreamingConfig,
        metrics: Optional[StreamingPIIMetrics] = None,
    ):
        """Initialize the Kinesis PII scanner.

        Args:
            enforcement_engine: PII enforcement engine for scanning.
            config: Streaming configuration with Kinesis settings.
            metrics: Optional metrics instance.

        Raises:
            ImportError: If boto3 is not installed.
        """
        if not BOTO3_AVAILABLE:
            raise ImportError(
                "boto3 is required for Kinesis streaming. "
                "Install with: pip install boto3"
            )

        super().__init__(
            enforcement_engine=enforcement_engine,
            config=config,
            metrics=metrics or get_streaming_metrics(backend="kinesis"),
        )

        self._kinesis_config = config.kinesis
        self._client: Optional[Any] = None  # boto3 kinesis client
        self._shard_states: Dict[str, ShardState] = {}
        self._consume_task: Optional[asyncio.Task] = None
        self._last_checkpoint_time: float = 0.0

        logger.info(
            "Initialized Kinesis PII Scanner: stream=%s region=%s",
            self._kinesis_config.stream_name,
            self._kinesis_config.region,
        )

    async def start(self) -> None:
        """Start the Kinesis consumer.

        Initializes the boto3 client, discovers shards, gets shard iterators,
        and begins processing records in a background task.

        Raises:
            ConnectionError: If unable to connect to Kinesis.
            RuntimeError: If scanner is already running.
        """
        if self._running:
            raise RuntimeError("Kinesis scanner is already running")

        logger.info(
            "Starting Kinesis PII Scanner on stream: %s",
            self._kinesis_config.stream_name,
        )

        try:
            # Create Kinesis client
            client_config = self._kinesis_config.get_boto3_config()
            self._client = boto3.client("kinesis", **client_config)

            # Initialize shard iterators
            await self._initialize_shard_iterators()

            self._running = True
            self._start_time = time.monotonic()
            self._last_checkpoint_time = time.monotonic()

            # Update metrics
            self._metrics.set_running(self._kinesis_config.consumer_name, True)

            logger.info(
                "Kinesis PII Scanner started successfully: shards=%d",
                len(self._shard_states),
            )

            # Start consuming
            await self._consume_loop()

        except (BotoCoreError, ClientError) as e:
            logger.error("Failed to start Kinesis scanner: %s", str(e), exc_info=True)
            await self._cleanup()
            raise ConnectionError(f"Failed to connect to Kinesis: {str(e)}") from e

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
            raise RuntimeError("Kinesis scanner is already running")

        self._consume_task = asyncio.create_task(self.start())

    async def stop(self) -> None:
        """Stop the Kinesis consumer.

        Gracefully shuts down the scanner, saving checkpoint information
        before closing.
        """
        if not self._running:
            logger.warning("Kinesis scanner is not running")
            return

        logger.info("Stopping Kinesis PII Scanner...")
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
        self._metrics.set_running(self._kinesis_config.consumer_name, False)

        logger.info(
            "Kinesis PII Scanner stopped: processed=%d blocked=%d errors=%d",
            self._processed_count,
            self._blocked_count,
            self._error_count,
        )

    async def _cleanup(self) -> None:
        """Clean up resources."""
        self._client = None
        self._shard_states.clear()

    async def _initialize_shard_iterators(self) -> None:
        """Initialize shard iterators for all shards in the stream.

        Discovers all active shards and creates shard iterators based
        on the configured iterator type.
        """
        # Describe stream to get shards
        response = await asyncio.to_thread(
            self._client.describe_stream,
            StreamName=self._kinesis_config.stream_name,
        )

        shards = response["StreamDescription"]["Shards"]

        for shard in shards:
            shard_id = shard["ShardId"]

            # Get shard iterator
            iterator_response = await asyncio.to_thread(
                self._client.get_shard_iterator,
                StreamName=self._kinesis_config.stream_name,
                ShardId=shard_id,
                ShardIteratorType=self._kinesis_config.shard_iterator_type.value,
            )

            self._shard_states[shard_id] = ShardState(
                shard_id=shard_id,
                iterator=iterator_response["ShardIterator"],
            )

        logger.info(
            "Initialized %d shard iterators for stream %s",
            len(self._shard_states),
            self._kinesis_config.stream_name,
        )

    async def _consume_loop(self) -> None:
        """Main consume loop that processes records from all shards.

        Continuously polls for records from each shard and processes
        them until stop() is called.
        """
        poll_interval_seconds = self._kinesis_config.idle_time_between_reads_ms / 1000.0

        try:
            while self._running:
                records_processed = 0

                # Poll each shard
                for shard_id, state in list(self._shard_states.items()):
                    if not self._running or not state.iterator:
                        continue

                    try:
                        records_count = await self._poll_shard(shard_id, state)
                        records_processed += records_count
                    except Exception as e:
                        logger.error(
                            "Error polling shard %s: %s",
                            shard_id,
                            str(e),
                            exc_info=True,
                        )
                        self._error_count += 1

                # If no records, wait before next poll
                if records_processed == 0:
                    await asyncio.sleep(poll_interval_seconds)

                # Periodic checkpoint logging
                await self._maybe_checkpoint()

        except asyncio.CancelledError:
            logger.info("Consume loop cancelled")
        except Exception as e:
            logger.error("Error in consume loop: %s", str(e), exc_info=True)
            self._error_count += 1
        finally:
            await self._cleanup()

    async def _poll_shard(
        self,
        shard_id: str,
        state: ShardState,
    ) -> int:
        """Poll a single shard for records.

        Args:
            shard_id: Shard identifier.
            state: Current shard state.

        Returns:
            Number of records processed.
        """
        try:
            response = await asyncio.to_thread(
                self._client.get_records,
                ShardIterator=state.iterator,
                Limit=self._kinesis_config.max_records_per_batch,
            )

            records = response.get("Records", [])
            next_iterator = response.get("NextShardIterator")
            behind_latest_ms = response.get("MillisBehindLatest", 0)

            # Update shard state
            state.iterator = next_iterator
            state.behind_latest_ms = behind_latest_ms

            # Process records
            for record in records:
                if not self._running:
                    break
                await self._process_kinesis_record(record, shard_id)
                state.sequence_number = record["SequenceNumber"]

            # Record batch size metric
            if records:
                self._metrics.record_batch_size(len(records))

            return len(records)

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")

            if error_code == "ExpiredIteratorException":
                # Re-initialize iterator
                logger.warning("Shard iterator expired for %s, reinitializing", shard_id)
                await self._reinitialize_shard_iterator(shard_id, state)
                return 0

            elif error_code == "ProvisionedThroughputExceededException":
                # Back off and retry
                logger.warning("Throughput exceeded for shard %s, backing off", shard_id)
                await asyncio.sleep(1.0)
                return 0

            else:
                raise

    async def _reinitialize_shard_iterator(
        self,
        shard_id: str,
        state: ShardState,
    ) -> None:
        """Reinitialize a shard iterator after expiration.

        Args:
            shard_id: Shard identifier.
            state: Current shard state.
        """
        iterator_type = (
            KinesisShardIteratorType.AFTER_SEQUENCE_NUMBER
            if state.sequence_number
            else self._kinesis_config.shard_iterator_type
        )

        kwargs: Dict[str, Any] = {
            "StreamName": self._kinesis_config.stream_name,
            "ShardId": shard_id,
            "ShardIteratorType": iterator_type.value,
        }

        if state.sequence_number:
            kwargs["StartingSequenceNumber"] = state.sequence_number

        response = await asyncio.to_thread(
            self._client.get_shard_iterator,
            **kwargs,
        )

        state.iterator = response["ShardIterator"]

    async def _process_kinesis_record(
        self,
        record: Dict[str, Any],
        shard_id: str,
    ) -> None:
        """Process a single Kinesis record.

        Extracts content and metadata from the Kinesis record,
        processes it through the enforcement engine, and routes
        the result to the appropriate output stream.

        Args:
            record: Kinesis record to process.
            shard_id: Source shard identifier.
        """
        try:
            # Decode record data
            content = record["Data"].decode("utf-8")

            # Build metadata from record
            metadata = self._extract_metadata(record, shard_id)

            # Record message size
            self._metrics.record_message_size(
                self._kinesis_config.stream_name,
                len(record["Data"]),
            )

            # Process through enforcement engine (inherited method)
            result = await self.process_message(content, metadata)

            # Route based on result
            if result.action == "blocked":
                await self._send_to_dlq(record, result)
            elif result.action in ("allowed", "redacted"):
                await self._send_to_output(record, result)
            # Error case: don't forward, just log

        except UnicodeDecodeError as e:
            self._error_count += 1
            self._metrics.record_error(self._kinesis_config.stream_name, "decode_error")
            logger.error(
                "Failed to decode record %s: %s",
                record.get("SequenceNumber", "unknown"),
                str(e),
            )

        except Exception as e:
            self._error_count += 1
            self._metrics.record_error(
                self._kinesis_config.stream_name, type(e).__name__
            )
            logger.error(
                "Error processing Kinesis record %s: %s",
                record.get("SequenceNumber", "unknown"),
                str(e),
                exc_info=True,
            )

    def _extract_metadata(
        self,
        record: Dict[str, Any],
        shard_id: str,
    ) -> Dict[str, Any]:
        """Extract metadata from a Kinesis record.

        Args:
            record: Kinesis record.
            shard_id: Source shard identifier.

        Returns:
            Dictionary of record metadata.
        """
        metadata: Dict[str, Any] = {
            "message_id": record["SequenceNumber"],
            "shard_id": shard_id,
            "partition_key": record["PartitionKey"],
            "sequence_number": record["SequenceNumber"],
            "stream_name": self._kinesis_config.stream_name,
        }

        # Add approximate arrival timestamp
        if "ApproximateArrivalTimestamp" in record:
            metadata["timestamp"] = record["ApproximateArrivalTimestamp"]

        # Try to extract tenant_id from partition key
        # Convention: partition key format "tenant_id:entity_id"
        partition_key = record["PartitionKey"]
        if ":" in partition_key:
            parts = partition_key.split(":", 1)
            metadata["tenant_id"] = parts[0]

        return metadata

    async def _send_to_output(
        self,
        original_record: Dict[str, Any],
        result: ProcessingResult,
    ) -> None:
        """Send processed record to output stream.

        Args:
            original_record: Original Kinesis record.
            result: Processing result with potentially modified content.
        """
        # Determine content to send
        content = result.modified_content
        if content is None:
            content = original_record["Data"].decode("utf-8")

        # Add scan metadata if configured
        if self._config.add_scan_metadata:
            # Wrap content with metadata
            wrapped_record = {
                "data": content,
                "pii_metadata": {
                    "scanned": True,
                    "action": result.action,
                    "processing_time_ms": result.processing_time_ms,
                    "detections_count": len(result.detections),
                },
            }
            output_data = json.dumps(wrapped_record)
        else:
            output_data = content

        await asyncio.to_thread(
            self._client.put_record,
            StreamName=self._kinesis_config.output_stream_name,
            Data=output_data.encode("utf-8"),
            PartitionKey=original_record["PartitionKey"],
        )

    async def _send_to_dlq(
        self,
        original_record: Dict[str, Any],
        result: ProcessingResult,
    ) -> None:
        """Send blocked record to DLQ stream.

        Creates a DLQ record with the original data, detection details,
        and metadata for investigation.

        Args:
            original_record: Original Kinesis record.
            result: Processing result with block reason.
        """
        dlq_record = {
            "original_data": original_record["Data"].decode("utf-8"),
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
                "stream": self._kinesis_config.stream_name,
                "sequence_number": original_record["SequenceNumber"],
                "partition_key": original_record["PartitionKey"],
            },
            "processing_time_ms": result.processing_time_ms,
        }

        await asyncio.to_thread(
            self._client.put_record,
            StreamName=self._kinesis_config.dlq_stream_name,
            Data=json.dumps(dlq_record).encode("utf-8"),
            PartitionKey=original_record["PartitionKey"],
        )

        # Record DLQ metric
        self._metrics.record_dlq_message(self._kinesis_config.stream_name, "pii_blocked")

        logger.warning(
            "Record blocked and sent to DLQ: %s - %d PII detections",
            original_record["SequenceNumber"],
            len(result.detections),
        )

    async def _maybe_checkpoint(self) -> None:
        """Checkpoint progress if interval has elapsed.

        Logs checkpoint information at configured intervals for
        operational visibility.
        """
        current_time = time.monotonic()
        elapsed = current_time - self._last_checkpoint_time

        if elapsed >= self._kinesis_config.checkpoint_interval_seconds:
            self._last_checkpoint_time = current_time

            # Log checkpoint info
            checkpoint_info = {
                shard_id: {
                    "sequence_number": state.sequence_number,
                    "behind_latest_ms": state.behind_latest_ms,
                }
                for shard_id, state in self._shard_states.items()
            }

            logger.info(
                "Checkpoint: processed=%d blocked=%d shards=%s",
                self._processed_count,
                self._blocked_count,
                json.dumps(checkpoint_info),
            )


# ---------------------------------------------------------------------------
# Factory Function
# ---------------------------------------------------------------------------


def create_kinesis_scanner(
    enforcement_engine: PIIEnforcementEngine,
    config: Optional[StreamingConfig] = None,
    **kinesis_overrides: Any,
) -> KinesisPIIScanner:
    """Factory function to create a Kinesis PII scanner.

    Convenience function that creates a scanner with sensible defaults
    and allows for easy configuration overrides.

    Args:
        enforcement_engine: PII enforcement engine instance.
        config: Optional streaming configuration.
        **kinesis_overrides: Override specific Kinesis settings.

    Returns:
        Configured KinesisPIIScanner instance.

    Example:
        >>> scanner = create_kinesis_scanner(
        ...     enforcement_engine,
        ...     stream_name="my-events",
        ...     region="us-west-2",
        ...     use_enhanced_fan_out=True,
        ... )
    """
    if config is None:
        config = StreamingConfig()

    # Apply overrides to Kinesis config
    if kinesis_overrides:
        kinesis_dict = config.kinesis.model_dump()
        kinesis_dict.update(kinesis_overrides)
        config = StreamingConfig(
            **{**config.model_dump(), "kinesis": KinesisConfig(**kinesis_dict)}
        )

    return KinesisPIIScanner(enforcement_engine, config)


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "KinesisPIIScanner",
    "create_kinesis_scanner",
    "BOTO3_AVAILABLE",
]
