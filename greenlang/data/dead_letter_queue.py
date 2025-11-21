"""
Dead Letter Queue (DLQ) for GreenLang data pipeline.

Handles failed records with quarantine, inspection, reprocessing, and analysis capabilities.
Ensures no data loss during pipeline failures.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from enum import Enum
import hashlib
import traceback
from pathlib import Path
import pickle
import asyncio
from collections import defaultdict
from greenlang.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class FailureReason(Enum):
    """Categorized failure reasons for analysis."""
    VALIDATION_ERROR = "validation_error"
    PARSING_ERROR = "parsing_error"
    TRANSFORMATION_ERROR = "transformation_error"
    SCHEMA_MISMATCH = "schema_mismatch"
    BUSINESS_RULE_VIOLATION = "business_rule_violation"
    TIMEOUT = "timeout"
    RESOURCE_UNAVAILABLE = "resource_unavailable"
    DUPLICATE_RECORD = "duplicate_record"
    UNKNOWN_ERROR = "unknown_error"


class ReprocessingStrategy(Enum):
    """Strategies for reprocessing failed records."""
    IMMEDIATE = "immediate"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    CONDITIONAL = "conditional"


@dataclass
class FailedRecord:
    """
    Container for failed record with metadata.

    Attributes:
        record_id: Unique identifier for the failed record
        original_data: The original data that failed processing
        failure_reason: Categorized reason for failure
        error_message: Detailed error message
        stack_trace: Full stack trace for debugging
        pipeline_stage: Stage where failure occurred
        timestamp: When the failure occurred
        retry_count: Number of retry attempts
        metadata: Additional context information
    """
    record_id: str
    original_data: Any
    failure_reason: FailureReason
    error_message: str
    stack_trace: str
    pipeline_stage: str
    timestamp: datetime
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_retry_timestamp: Optional[datetime] = None
    reprocessing_strategy: ReprocessingStrategy = ReprocessingStrategy.EXPONENTIAL_BACKOFF

    def can_retry(self) -> bool:
        """Check if record can be retried."""
        return self.retry_count < self.max_retries

    def should_retry_now(self) -> bool:
        """Check if record should be retried now based on strategy."""
        if not self.can_retry():
            return False

        if self.reprocessing_strategy == ReprocessingStrategy.IMMEDIATE:
            return True

        if self.reprocessing_strategy == ReprocessingStrategy.EXPONENTIAL_BACKOFF:
            if not self.last_retry_timestamp:
                return True
            # Exponential backoff: 2^retry_count minutes
            wait_time = timedelta(minutes=(2 ** self.retry_count))
            return DeterministicClock.now() >= self.last_retry_timestamp + wait_time

        if self.reprocessing_strategy == ReprocessingStrategy.MANUAL:
            return False

        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "record_id": self.record_id,
            "original_data": self.original_data,
            "failure_reason": self.failure_reason.value,
            "error_message": self.error_message,
            "stack_trace": self.stack_trace,
            "pipeline_stage": self.pipeline_stage,
            "timestamp": self.timestamp.isoformat(),
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "metadata": self.metadata,
            "last_retry_timestamp": self.last_retry_timestamp.isoformat() if self.last_retry_timestamp else None,
            "reprocessing_strategy": self.reprocessing_strategy.value
        }


class DeadLetterQueue:
    """
    Dead Letter Queue for failed pipeline records.

    Features:
    - Quarantine failed records with full context
    - Automatic categorization of failures
    - Configurable reprocessing strategies
    - Analytics and monitoring
    - Persistence for recovery
    - Integration with alerting systems
    """

    def __init__(
        self,
        max_queue_size: int = 10000,
        persistence_path: Optional[Path] = None,
        alert_threshold: int = 100,
        auto_retry: bool = True
    ):
        """
        Initialize Dead Letter Queue.

        Args:
            max_queue_size: Maximum number of records to keep in queue
            persistence_path: Optional path for persistent storage
            alert_threshold: Number of failures before alerting
            auto_retry: Whether to automatically retry failed records
        """
        self.max_queue_size = max_queue_size
        self.persistence_path = persistence_path
        self.alert_threshold = alert_threshold
        self.auto_retry = auto_retry

        # In-memory storage
        self._queue: Dict[str, FailedRecord] = {}
        self._failure_stats: Dict[str, int] = defaultdict(int)
        self._stage_stats: Dict[str, int] = defaultdict(int)

        # Reprocessing handlers
        self._reprocessing_handlers: Dict[str, Callable] = {}

        # Load persisted queue if available
        if persistence_path:
            self._load_persisted_queue()

    def quarantine_record(
        self,
        record: Any,
        error: Exception,
        pipeline_stage: str,
        metadata: Optional[Dict[str, Any]] = None,
        reprocessing_strategy: ReprocessingStrategy = ReprocessingStrategy.EXPONENTIAL_BACKOFF
    ) -> str:
        """
        Quarantine a failed record to the DLQ.

        Args:
            record: The record that failed processing
            error: The exception that caused the failure
            pipeline_stage: The stage where failure occurred
            metadata: Additional context information
            reprocessing_strategy: Strategy for reprocessing

        Returns:
            Record ID for tracking

        Example:
            try:
                process_record(data)
            except Exception as e:
                dlq.quarantine_record(
                    record=data,
                    error=e,
                    pipeline_stage="transformation",
                    metadata={"source": "sap", "batch_id": "12345"}
                )
        """
        # Generate unique record ID
        record_id = self._generate_record_id(record, pipeline_stage)

        # Categorize failure
        failure_reason = self._categorize_failure(error)

        # Create failed record
        failed_record = FailedRecord(
            record_id=record_id,
            original_data=record,
            failure_reason=failure_reason,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            pipeline_stage=pipeline_stage,
            timestamp=DeterministicClock.now(),
            metadata=metadata or {},
            reprocessing_strategy=reprocessing_strategy
        )

        # Check if record already exists (retry case)
        if record_id in self._queue:
            existing = self._queue[record_id]
            failed_record.retry_count = existing.retry_count + 1
            failed_record.max_retries = existing.max_retries

        # Add to queue
        self._queue[record_id] = failed_record

        # Update statistics
        self._failure_stats[failure_reason.value] += 1
        self._stage_stats[pipeline_stage] += 1

        # Check queue size
        if len(self._queue) >= self.max_queue_size:
            self._evict_oldest_records()

        # Check alert threshold
        if len(self._queue) >= self.alert_threshold:
            self._trigger_alert()

        # Persist if configured
        if self.persistence_path:
            self._persist_queue()

        logger.error(
            f"Record quarantined to DLQ: {record_id} "
            f"(Stage: {pipeline_stage}, Reason: {failure_reason.value})"
        )

        return record_id

    def get_failed_records(
        self,
        failure_reason: Optional[FailureReason] = None,
        pipeline_stage: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[FailedRecord]:
        """
        Retrieve failed records with filters.

        Args:
            failure_reason: Filter by failure reason
            pipeline_stage: Filter by pipeline stage
            start_time: Filter by timestamp (after)
            end_time: Filter by timestamp (before)
            limit: Maximum number of records to return

        Returns:
            List of failed records matching criteria

        Example:
            # Get all validation errors from the last hour
            failed = dlq.get_failed_records(
                failure_reason=FailureReason.VALIDATION_ERROR,
                start_time=DeterministicClock.now() - timedelta(hours=1)
            )
        """
        records = list(self._queue.values())

        # Apply filters
        if failure_reason:
            records = [r for r in records if r.failure_reason == failure_reason]

        if pipeline_stage:
            records = [r for r in records if r.pipeline_stage == pipeline_stage]

        if start_time:
            records = [r for r in records if r.timestamp >= start_time]

        if end_time:
            records = [r for r in records if r.timestamp <= end_time]

        # Sort by timestamp (most recent first)
        records.sort(key=lambda x: x.timestamp, reverse=True)

        return records[:limit]

    def reprocess(
        self,
        record_id: Optional[str] = None,
        failure_reason: Optional[FailureReason] = None,
        pipeline_stage: Optional[str] = None,
        max_records: int = 10,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Reprocess failed records.

        Args:
            record_id: Specific record to reprocess
            failure_reason: Reprocess all records with this failure reason
            pipeline_stage: Reprocess all records from this stage
            max_records: Maximum records to reprocess in one batch
            force: Force reprocessing even if retry limit exceeded

        Returns:
            Dictionary with reprocessing results

        Example:
            # Reprocess specific record
            result = dlq.reprocess(record_id="abc123")

            # Reprocess all validation errors
            result = dlq.reprocess(
                failure_reason=FailureReason.VALIDATION_ERROR,
                max_records=50
            )
        """
        results = {
            "processed": 0,
            "succeeded": 0,
            "failed": 0,
            "skipped": 0,
            "details": []
        }

        # Get records to reprocess
        if record_id:
            records_to_process = [self._queue.get(record_id)] if record_id in self._queue else []
        else:
            records_to_process = self.get_failed_records(
                failure_reason=failure_reason,
                pipeline_stage=pipeline_stage,
                limit=max_records
            )

        for record in records_to_process:
            if not record:
                continue

            results["processed"] += 1

            # Check if can retry
            if not force and not record.can_retry():
                results["skipped"] += 1
                results["details"].append({
                    "record_id": record.record_id,
                    "status": "skipped",
                    "reason": "max_retries_exceeded"
                })
                continue

            # Check if should retry now
            if not force and not record.should_retry_now():
                results["skipped"] += 1
                results["details"].append({
                    "record_id": record.record_id,
                    "status": "skipped",
                    "reason": "retry_not_due"
                })
                continue

            # Get reprocessing handler
            handler = self._reprocessing_handlers.get(record.pipeline_stage)
            if not handler:
                logger.warning(f"No reprocessing handler for stage: {record.pipeline_stage}")
                results["failed"] += 1
                results["details"].append({
                    "record_id": record.record_id,
                    "status": "failed",
                    "reason": "no_handler"
                })
                continue

            try:
                # Attempt reprocessing
                handler(record.original_data, record.metadata)

                # Success - remove from queue
                del self._queue[record.record_id]
                results["succeeded"] += 1
                results["details"].append({
                    "record_id": record.record_id,
                    "status": "succeeded",
                    "retry_count": record.retry_count
                })

                logger.info(f"Successfully reprocessed record: {record.record_id}")

            except Exception as e:
                # Failed - update retry count
                record.retry_count += 1
                record.last_retry_timestamp = DeterministicClock.now()
                self._queue[record.record_id] = record

                results["failed"] += 1
                results["details"].append({
                    "record_id": record.record_id,
                    "status": "failed",
                    "error": str(e),
                    "retry_count": record.retry_count
                })

                logger.error(f"Failed to reprocess record {record.record_id}: {e}")

        # Persist changes
        if self.persistence_path:
            self._persist_queue()

        return results

    async def async_reprocess(
        self,
        batch_size: int = 10,
        parallel_workers: int = 3
    ) -> Dict[str, Any]:
        """
        Asynchronously reprocess failed records in parallel.

        Args:
            batch_size: Number of records per batch
            parallel_workers: Number of parallel workers

        Returns:
            Aggregated reprocessing results
        """
        # Get records ready for retry
        records = [
            r for r in self._queue.values()
            if r.should_retry_now()
        ][:batch_size * parallel_workers]

        # Split into batches
        batches = [
            records[i:i + batch_size]
            for i in range(0, len(records), batch_size)
        ]

        # Process batches in parallel
        tasks = [
            self._async_reprocess_batch(batch)
            for batch in batches
        ]

        results = await asyncio.gather(*tasks)

        # Aggregate results
        total_results = {
            "processed": sum(r["processed"] for r in results),
            "succeeded": sum(r["succeeded"] for r in results),
            "failed": sum(r["failed"] for r in results),
            "skipped": sum(r["skipped"] for r in results)
        }

        return total_results

    def register_reprocessing_handler(
        self,
        pipeline_stage: str,
        handler: Callable
    ):
        """
        Register a handler for reprocessing failed records from a specific stage.

        Args:
            pipeline_stage: Pipeline stage name
            handler: Function to handle reprocessing

        Example:
            def reprocess_transformation(data, metadata):
                # Custom reprocessing logic
                return transform_data(data, strict=False)

            dlq.register_reprocessing_handler("transformation", reprocess_transformation)
        """
        self._reprocessing_handlers[pipeline_stage] = handler
        logger.info(f"Registered reprocessing handler for stage: {pipeline_stage}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get DLQ statistics for monitoring.

        Returns:
            Dictionary with queue statistics
        """
        records = list(self._queue.values())

        if not records:
            return {
                "total_records": 0,
                "by_failure_reason": {},
                "by_pipeline_stage": {},
                "retry_statistics": {},
                "oldest_record": None,
                "newest_record": None
            }

        return {
            "total_records": len(records),
            "by_failure_reason": dict(self._failure_stats),
            "by_pipeline_stage": dict(self._stage_stats),
            "retry_statistics": {
                "awaiting_retry": len([r for r in records if r.can_retry()]),
                "max_retries_exceeded": len([r for r in records if not r.can_retry()]),
                "average_retry_count": sum(r.retry_count for r in records) / len(records)
            },
            "oldest_record": min(records, key=lambda x: x.timestamp).timestamp.isoformat(),
            "newest_record": max(records, key=lambda x: x.timestamp).timestamp.isoformat(),
            "queue_capacity": f"{len(records)}/{self.max_queue_size}"
        }

    def clear_queue(
        self,
        older_than: Optional[timedelta] = None,
        failure_reason: Optional[FailureReason] = None
    ) -> int:
        """
        Clear records from the queue.

        Args:
            older_than: Remove records older than this duration
            failure_reason: Remove only records with this failure reason

        Returns:
            Number of records removed
        """
        initial_count = len(self._queue)

        if older_than:
            cutoff_time = DeterministicClock.now() - older_than
            self._queue = {
                k: v for k, v in self._queue.items()
                if v.timestamp > cutoff_time
            }

        if failure_reason:
            self._queue = {
                k: v for k, v in self._queue.items()
                if v.failure_reason != failure_reason
            }

        removed = initial_count - len(self._queue)

        if removed > 0 and self.persistence_path:
            self._persist_queue()

        logger.info(f"Cleared {removed} records from DLQ")
        return removed

    def export_records(
        self,
        output_path: Path,
        format: str = "json"
    ) -> int:
        """
        Export DLQ records for analysis.

        Args:
            output_path: Path to export file
            format: Export format (json, csv, pickle)

        Returns:
            Number of records exported
        """
        records = list(self._queue.values())

        if format == "json":
            with open(output_path, 'w') as f:
                json.dump(
                    [r.to_dict() for r in records],
                    f,
                    indent=2,
                    default=str
                )
        elif format == "pickle":
            with open(output_path, 'wb') as f:
                pickle.dump(records, f)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Exported {len(records)} records to {output_path}")
        return len(records)

    def _generate_record_id(self, record: Any, stage: str) -> str:
        """Generate unique ID for record."""
        content = f"{str(record)}_{stage}_{DeterministicClock.now().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _categorize_failure(self, error: Exception) -> FailureReason:
        """Categorize failure based on exception type."""
        error_str = str(error).lower()
        error_type = type(error).__name__

        if "validation" in error_str or "ValidationError" in error_type:
            return FailureReason.VALIDATION_ERROR
        elif "parse" in error_str or "ParseError" in error_type:
            return FailureReason.PARSING_ERROR
        elif "schema" in error_str:
            return FailureReason.SCHEMA_MISMATCH
        elif "timeout" in error_str or "TimeoutError" in error_type:
            return FailureReason.TIMEOUT
        elif "duplicate" in error_str:
            return FailureReason.DUPLICATE_RECORD
        elif "transform" in error_str:
            return FailureReason.TRANSFORMATION_ERROR
        elif "business rule" in error_str or "constraint" in error_str:
            return FailureReason.BUSINESS_RULE_VIOLATION
        elif "connection" in error_str or "unavailable" in error_str:
            return FailureReason.RESOURCE_UNAVAILABLE
        else:
            return FailureReason.UNKNOWN_ERROR

    def _evict_oldest_records(self):
        """Evict oldest records when queue is full."""
        # Sort by timestamp and remove oldest 10%
        sorted_records = sorted(
            self._queue.items(),
            key=lambda x: x[1].timestamp
        )
        records_to_remove = int(self.max_queue_size * 0.1)

        for record_id, _ in sorted_records[:records_to_remove]:
            del self._queue[record_id]

        logger.warning(f"Evicted {records_to_remove} oldest records from DLQ")

    def _trigger_alert(self):
        """Trigger alert when threshold exceeded."""
        logger.critical(
            f"DLQ alert threshold exceeded: {len(self._queue)} records in queue "
            f"(threshold: {self.alert_threshold})"
        )
        # Here you would integrate with alerting system (PagerDuty, SNS, etc.)

    def _persist_queue(self):
        """Persist queue to disk."""
        if not self.persistence_path:
            return

        self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.persistence_path, 'wb') as f:
            pickle.dump(self._queue, f)

    def _load_persisted_queue(self):
        """Load persisted queue from disk."""
        if self.persistence_path and self.persistence_path.exists():
            try:
                with open(self.persistence_path, 'rb') as f:
                    self._queue = pickle.load(f)
                logger.info(f"Loaded {len(self._queue)} records from persisted DLQ")
            except Exception as e:
                logger.error(f"Failed to load persisted DLQ: {e}")

    async def _async_reprocess_batch(
        self,
        batch: List[FailedRecord]
    ) -> Dict[str, Any]:
        """Asynchronously reprocess a batch of records."""
        results = {
            "processed": 0,
            "succeeded": 0,
            "failed": 0,
            "skipped": 0
        }

        for record in batch:
            results["processed"] += 1
            handler = self._reprocessing_handlers.get(record.pipeline_stage)

            if not handler:
                results["skipped"] += 1
                continue

            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(record.original_data, record.metadata)
                else:
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        handler,
                        record.original_data,
                        record.metadata
                    )

                del self._queue[record.record_id]
                results["succeeded"] += 1

            except Exception:
                record.retry_count += 1
                record.last_retry_timestamp = DeterministicClock.now()
                results["failed"] += 1

        return results


# Global DLQ instance for easy access
_global_dlq: Optional[DeadLetterQueue] = None


def get_dlq() -> DeadLetterQueue:
    """Get global DLQ instance."""
    global _global_dlq
    if _global_dlq is None:
        _global_dlq = DeadLetterQueue(
            persistence_path=Path("data/dlq/queue.pkl")
        )
    return _global_dlq