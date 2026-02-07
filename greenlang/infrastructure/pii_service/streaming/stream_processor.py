# -*- coding: utf-8 -*-
"""
Base Stream Processor - SEC-011 Streaming PII Scanner

Abstract base class for stream processors that scan messages for PII.
Provides common processing logic, metrics recording, and error handling
that Kafka and Kinesis scanners inherit from.

Architecture:
    BaseStreamProcessor (abstract)
    ├── process_message() - Common PII scanning logic
    ├── get_stats() - Processing statistics
    └── abstract start()/stop() - Lifecycle management

    KafkaPIIScanner extends BaseStreamProcessor
    KinesisPIIScanner extends BaseStreamProcessor

Processing Flow:
    1. Receive message from stream
    2. Decode message content
    3. Extract metadata (headers, partition, offset, etc.)
    4. Scan for PII using enforcement engine
    5. Apply enforcement action (allow/redact/block)
    6. Record metrics and audit trail
    7. Route to output topic or DLQ

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-011 PII Detection/Redaction Enhancements
Status: Production Ready
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from greenlang.infrastructure.pii_service.streaming.config import (
    EnforcementMode,
    StreamingConfig,
)
from greenlang.infrastructure.pii_service.streaming.metrics import (
    StreamingPIIMetrics,
    get_streaming_metrics,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class PIIDetection(BaseModel):
    """A single PII detection within a message.

    Represents one instance of detected PII with type, location,
    and confidence information.

    Attributes:
        pii_type: Type of PII detected (e.g., "ssn", "email").
        confidence: Detection confidence score (0.0 to 1.0).
        start: Start character offset in the content.
        end: End character offset in the content.
        context: Surrounding context (redacted for safety).
        value_hash: SHA-256 hash of the detected value (never raw).
    """

    model_config = ConfigDict(frozen=True)

    pii_type: str = Field(..., description="Type of PII detected")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Detection confidence (0-1)",
    )
    start: int = Field(..., ge=0, description="Start offset in content")
    end: int = Field(..., ge=0, description="End offset in content")
    context: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Surrounding context (redacted)",
    )
    value_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash of detected value",
    )


class ProcessingResult(BaseModel):
    """Result of processing a single stream message.

    Contains the outcome of PII scanning and enforcement for one message,
    including what action was taken and any detections found.

    Attributes:
        message_id: Unique identifier for the message.
        action: Action taken (allowed, redacted, blocked).
        original_size: Size of original message in bytes.
        processed_size: Size of processed message in bytes.
        detections: List of PII detections found.
        processing_time_ms: Time spent processing in milliseconds.
        modified_content: The modified content (if redacted).
        error: Error message if processing failed.
        timestamp: When processing completed.

    Example:
        >>> result = ProcessingResult(
        ...     message_id="topic:0:12345",
        ...     action="redacted",
        ...     original_size=1024,
        ...     processed_size=1020,
        ...     detections=[PIIDetection(pii_type="email", confidence=0.95, start=10, end=30)],
        ...     processing_time_ms=5.2,
        ... )
    """

    model_config = ConfigDict(frozen=False)

    message_id: str = Field(
        ...,
        description="Unique message identifier",
    )
    action: str = Field(
        ...,
        pattern=r"^(allowed|redacted|blocked|error)$",
        description="Action taken on message",
    )
    original_size: int = Field(
        ...,
        ge=0,
        description="Original message size (bytes)",
    )
    processed_size: int = Field(
        ...,
        ge=0,
        description="Processed message size (bytes)",
    )
    detections: List[PIIDetection] = Field(
        default_factory=list,
        description="PII detections found",
    )
    processing_time_ms: float = Field(
        ...,
        ge=0,
        description="Processing time (ms)",
    )
    modified_content: Optional[str] = Field(
        default=None,
        description="Modified content after redaction",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if processing failed",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Processing timestamp",
    )

    def to_dlq_record(self) -> Dict[str, Any]:
        """Convert to a DLQ record format.

        Returns:
            Dictionary suitable for DLQ message payload.
        """
        return {
            "message_id": self.message_id,
            "reason": "PII_BLOCKED",
            "detections": [
                {
                    "pii_type": d.pii_type,
                    "confidence": d.confidence,
                    "start": d.start,
                    "end": d.end,
                }
                for d in self.detections
            ],
            "timestamp": self.timestamp.isoformat(),
            "processing_time_ms": self.processing_time_ms,
        }


class ProcessingStats(BaseModel):
    """Statistics for stream processing.

    Aggregated statistics about stream processing performance
    and outcomes.

    Attributes:
        processed: Total messages processed.
        allowed: Messages allowed through unchanged.
        redacted: Messages redacted and forwarded.
        blocked: Messages blocked and sent to DLQ.
        errors: Processing errors encountered.
        total_detections: Total PII detections across all messages.
        avg_processing_time_ms: Average processing time.
        uptime_seconds: Scanner uptime in seconds.
    """

    model_config = ConfigDict(frozen=True)

    processed: int = Field(default=0, description="Total processed")
    allowed: int = Field(default=0, description="Messages allowed")
    redacted: int = Field(default=0, description="Messages redacted")
    blocked: int = Field(default=0, description="Messages blocked")
    errors: int = Field(default=0, description="Processing errors")
    total_detections: int = Field(default=0, description="Total PII detections")
    avg_processing_time_ms: float = Field(default=0.0, description="Avg processing time")
    uptime_seconds: float = Field(default=0.0, description="Scanner uptime")


# ---------------------------------------------------------------------------
# Mock Enforcement Engine Interface
# ---------------------------------------------------------------------------


class EnforcementResult(BaseModel):
    """Result from the enforcement engine.

    Represents the outcome of PII enforcement on content.
    """

    model_config = ConfigDict(frozen=False)

    blocked: bool = Field(default=False, description="Whether content was blocked")
    modified_content: Optional[str] = Field(
        default=None,
        description="Modified content after redaction",
    )
    detections: List[PIIDetection] = Field(
        default_factory=list,
        description="PII detections found",
    )


class EnforcementContext(BaseModel):
    """Context for enforcement decisions.

    Provides context about where content came from for
    policy-based enforcement decisions.
    """

    model_config = ConfigDict(frozen=False, extra="allow")

    context_type: str = Field(..., description="Type of context")
    tenant_id: str = Field(default="default", description="Tenant identifier")
    source: Optional[str] = Field(default=None, description="Source system")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class PIIEnforcementEngine:
    """Interface for PII enforcement engine.

    This is a simplified interface that matches the enforcement engine
    defined in SEC-011. The actual implementation would be imported from
    the enforcement module.
    """

    async def enforce(
        self,
        content: str,
        context: EnforcementContext,
    ) -> EnforcementResult:
        """Enforce PII policies on content.

        Args:
            content: Content to scan and potentially modify.
            context: Enforcement context for policy decisions.

        Returns:
            EnforcementResult with action taken and detections.
        """
        # This is a stub - actual implementation would scan content
        # and apply enforcement policies
        return EnforcementResult(
            blocked=False,
            modified_content=content,
            detections=[],
        )


# ---------------------------------------------------------------------------
# Base Stream Processor
# ---------------------------------------------------------------------------


class BaseStreamProcessor(ABC):
    """Abstract base class for stream processors.

    Provides common processing logic, metrics recording, and statistics
    tracking for Kafka and Kinesis PII scanners.

    This class handles:
        - Message scanning using the enforcement engine
        - Action determination based on configuration
        - Metrics recording for observability
        - Statistics aggregation
        - Error handling and recovery

    Subclasses must implement:
        - start(): Begin consuming from the stream
        - stop(): Stop consuming and clean up

    Attributes:
        config: Streaming configuration.
        metrics: Prometheus metrics instance.

    Example:
        >>> class MyScanner(BaseStreamProcessor):
        ...     async def start(self):
        ...         # Start consuming
        ...         pass
        ...     async def stop(self):
        ...         # Stop consuming
        ...         pass
        ...
        >>> scanner = MyScanner(enforcement_engine, config)
        >>> await scanner.start()
    """

    def __init__(
        self,
        enforcement_engine: PIIEnforcementEngine,
        config: StreamingConfig,
        metrics: Optional[StreamingPIIMetrics] = None,
    ):
        """Initialize the stream processor.

        Args:
            enforcement_engine: PII enforcement engine for scanning.
            config: Streaming configuration.
            metrics: Optional metrics instance (uses global if not provided).
        """
        self._enforcement = enforcement_engine
        self._config = config
        self._metrics = metrics or get_streaming_metrics()
        self._running = False
        self._start_time: Optional[float] = None

        # Statistics counters
        self._processed_count = 0
        self._allowed_count = 0
        self._redacted_count = 0
        self._blocked_count = 0
        self._error_count = 0
        self._total_detections = 0
        self._total_processing_time_ms = 0.0

        logger.info(
            "Initialized stream processor: enforcement_mode=%s min_confidence=%s",
            config.enforcement_mode.value,
            config.min_confidence,
        )

    @property
    def is_running(self) -> bool:
        """Check if the processor is currently running."""
        return self._running

    async def process_message(
        self,
        content: str,
        metadata: Dict[str, Any],
    ) -> ProcessingResult:
        """Process a single message through PII enforcement.

        This is the core processing method that:
        1. Creates enforcement context from metadata
        2. Calls the enforcement engine to scan content
        3. Determines the action based on results and config
        4. Records metrics and updates statistics
        5. Returns the processing result

        Args:
            content: Message content to scan.
            metadata: Message metadata (topic, partition, offset, etc.).

        Returns:
            ProcessingResult with action taken and detections.

        Example:
            >>> result = await processor.process_message(
            ...     content='Contact john@email.com for details',
            ...     metadata={
            ...         'message_id': 'topic:0:12345',
            ...         'topic': 'events',
            ...         'tenant_id': 'acme-corp',
            ...     }
            ... )
            >>> print(result.action)  # 'allowed', 'redacted', or 'blocked'
        """
        start_time = time.monotonic()
        message_id = metadata.get("message_id", str(uuid4()))
        topic = metadata.get("topic", "unknown")

        try:
            # Create enforcement context
            context = EnforcementContext(
                context_type="streaming",
                tenant_id=metadata.get("tenant_id", "default"),
                source=metadata.get("source"),
                metadata=metadata,
            )

            # Call enforcement engine
            enforcement_result = await self._enforcement.enforce(content, context)

            # Calculate processing time
            processing_time_ms = (time.monotonic() - start_time) * 1000

            # Determine action based on enforcement result and config
            action, modified_content = self._determine_action(
                content=content,
                enforcement_result=enforcement_result,
            )

            # Convert detections to our model format
            detections = self._convert_detections(enforcement_result.detections)

            # Update statistics
            self._update_stats(action, len(detections), processing_time_ms)

            # Record metrics
            self._record_metrics(action, topic, detections, processing_time_ms)

            return ProcessingResult(
                message_id=message_id,
                action=action,
                original_size=len(content.encode("utf-8")),
                processed_size=len(modified_content.encode("utf-8")) if modified_content else 0,
                detections=detections,
                processing_time_ms=processing_time_ms,
                modified_content=modified_content,
            )

        except Exception as e:
            processing_time_ms = (time.monotonic() - start_time) * 1000
            self._error_count += 1
            self._metrics.record_error(topic, type(e).__name__)
            logger.error(
                "Error processing message %s: %s",
                message_id,
                str(e),
                exc_info=True,
            )

            return ProcessingResult(
                message_id=message_id,
                action="error",
                original_size=len(content.encode("utf-8")),
                processed_size=0,
                detections=[],
                processing_time_ms=processing_time_ms,
                error=str(e),
            )

    def _determine_action(
        self,
        content: str,
        enforcement_result: EnforcementResult,
    ) -> tuple[str, Optional[str]]:
        """Determine the action to take based on enforcement result.

        Args:
            content: Original content.
            enforcement_result: Result from enforcement engine.

        Returns:
            Tuple of (action, modified_content).
        """
        # If enforcement says block, always block (regardless of config mode)
        if enforcement_result.blocked:
            return "blocked", None

        # Check enforcement mode
        mode = self._config.enforcement_mode

        if mode == EnforcementMode.ALLOW:
            # Allow mode: just log, pass through unchanged
            return "allowed", content

        elif mode == EnforcementMode.REDACT:
            # Redact mode: use modified content if available
            if enforcement_result.modified_content and enforcement_result.modified_content != content:
                return "redacted", enforcement_result.modified_content
            return "allowed", content

        elif mode == EnforcementMode.BLOCK:
            # Block mode: block if any high-confidence detections
            if enforcement_result.detections:
                high_confidence = any(
                    d.confidence >= self._config.min_confidence
                    for d in enforcement_result.detections
                )
                if high_confidence:
                    return "blocked", None
            return "allowed", content

        return "allowed", content

    def _convert_detections(
        self,
        detections: List[Any],
    ) -> List[PIIDetection]:
        """Convert enforcement detections to our model format.

        Args:
            detections: Detections from enforcement engine.

        Returns:
            List of PIIDetection objects.
        """
        result = []
        for d in detections:
            if isinstance(d, PIIDetection):
                result.append(d)
            elif hasattr(d, "pii_type") and hasattr(d, "confidence"):
                result.append(
                    PIIDetection(
                        pii_type=getattr(d, "pii_type", "unknown"),
                        confidence=getattr(d, "confidence", 0.0),
                        start=getattr(d, "start", 0),
                        end=getattr(d, "end", 0),
                        context=getattr(d, "context", None),
                        value_hash=getattr(d, "value_hash", None),
                    )
                )
        return result

    def _update_stats(
        self,
        action: str,
        detection_count: int,
        processing_time_ms: float,
    ) -> None:
        """Update internal statistics.

        Args:
            action: Action taken on message.
            detection_count: Number of PII detections.
            processing_time_ms: Processing time in milliseconds.
        """
        self._processed_count += 1
        self._total_detections += detection_count
        self._total_processing_time_ms += processing_time_ms

        if action == "allowed":
            self._allowed_count += 1
        elif action == "redacted":
            self._redacted_count += 1
        elif action == "blocked":
            self._blocked_count += 1

    def _record_metrics(
        self,
        action: str,
        topic: str,
        detections: List[PIIDetection],
        processing_time_ms: float,
    ) -> None:
        """Record Prometheus metrics.

        Args:
            action: Action taken on message.
            topic: Source topic name.
            detections: PII detections found.
            processing_time_ms: Processing time in milliseconds.
        """
        # Record processed message
        self._metrics.record_processed(topic, action)

        # Record processing time
        self._metrics.record_processing_time(processing_time_ms)

        # Record blocked messages with PII type
        if action == "blocked" and detections:
            # Use the highest confidence detection's type
            primary_type = max(detections, key=lambda d: d.confidence).pii_type
            self._metrics.record_blocked(topic, primary_type)

        # Record individual detections
        for detection in detections:
            self._metrics.record_detection(topic, detection.pii_type)

    def get_stats(self) -> ProcessingStats:
        """Get current processing statistics.

        Returns:
            ProcessingStats with current counters.

        Example:
            >>> stats = processor.get_stats()
            >>> print(f"Processed: {stats.processed}, Blocked: {stats.blocked}")
        """
        uptime = 0.0
        if self._start_time:
            uptime = time.monotonic() - self._start_time

        avg_time = 0.0
        if self._processed_count > 0:
            avg_time = self._total_processing_time_ms / self._processed_count

        return ProcessingStats(
            processed=self._processed_count,
            allowed=self._allowed_count,
            redacted=self._redacted_count,
            blocked=self._blocked_count,
            errors=self._error_count,
            total_detections=self._total_detections,
            avg_processing_time_ms=avg_time,
            uptime_seconds=uptime,
        )

    def reset_stats(self) -> None:
        """Reset all statistics counters."""
        self._processed_count = 0
        self._allowed_count = 0
        self._redacted_count = 0
        self._blocked_count = 0
        self._error_count = 0
        self._total_detections = 0
        self._total_processing_time_ms = 0.0
        logger.info("Statistics reset")

    @abstractmethod
    async def start(self) -> None:
        """Start the stream processor.

        Subclasses must implement this to begin consuming from their
        respective streaming platform.

        Raises:
            ConnectionError: If unable to connect to the stream.
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the stream processor.

        Subclasses must implement this to gracefully stop consuming
        and clean up resources.
        """
        pass


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "PIIDetection",
    "ProcessingResult",
    "ProcessingStats",
    "EnforcementResult",
    "EnforcementContext",
    "PIIEnforcementEngine",
    "BaseStreamProcessor",
]
