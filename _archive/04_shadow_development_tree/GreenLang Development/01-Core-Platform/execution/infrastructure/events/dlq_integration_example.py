"""
DLQ Handler Integration Example for Process Heat Agents

Demonstrates how to integrate the DeadLetterQueueHandler with Process Heat
agent pipelines, including:
- Kafka DLQ topic routing
- Redis-based retry tracking
- Exponential backoff with transient/permanent error categorization
- Batch reprocessing
- Prometheus metrics export

This is a reference implementation for backend teams.

Example:
    >>> dlq_manager = ProcessHeatDLQManager(config)
    >>> await dlq_manager.start()
    >>> await dlq_manager.handle_failed_event(event, error)
    >>> stats = await dlq_manager.get_metrics()
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

from greenlang.infrastructure.events.dlq_handler import (
    DeadLetterQueueHandler,
    DLQHandlerConfig,
    DLQMessage,
    DLQStats,
    ErrorCategory,
    DLQMessageStatus,
)

logger = logging.getLogger(__name__)


class ProcessHeatDLQManager:
    """
    Manages dead letter queues for Process Heat agents.

    Integrates DLQ handling with agent pipelines, providing:
    - Automatic DLQ routing based on queue names
    - Error categorization (transient vs permanent)
    - Retry policy configuration per queue
    - Metrics collection and exporting

    Example:
        >>> config = DLQHandlerConfig(
        ...     kafka_brokers=['kafka:9092'],
        ...     redis_url='redis://redis:6379',
        ...     max_retries=3
        ... )
        >>> manager = ProcessHeatDLQManager(config)
        >>> await manager.start()
    """

    def __init__(self, config: DLQHandlerConfig):
        """Initialize DLQ manager."""
        self.config = config
        self.dlq_handler = DeadLetterQueueHandler(config)

        # Queue-specific configurations
        self.queue_configs: Dict[str, Dict[str, Any]] = {}

        # Metrics
        self.metrics = {
            "total_dlq_messages": 0,
            "total_processed": 0,
            "total_escalated": 0,
            "queues": {}
        }

    async def start(self) -> None:
        """Start the DLQ manager."""
        await self.dlq_handler.start()

        # Set up alert callback
        self.dlq_handler.add_alert_callback(self._on_threshold_alert)

        logger.info("Process Heat DLQ Manager started")

    async def stop(self) -> None:
        """Stop the DLQ manager."""
        await self.dlq_handler.stop()
        logger.info("Process Heat DLQ Manager stopped")

    def configure_queue(
        self,
        queue_name: str,
        max_retries: int = 3,
        retry_delay_seconds: int = 60,
        transient_errors: Optional[List[str]] = None,
        permanent_errors: Optional[List[str]] = None
    ) -> None:
        """
        Configure DLQ handling for a specific queue.

        Args:
            queue_name: Queue/topic name (e.g., 'heat-calculations')
            max_retries: Maximum retry attempts
            retry_delay_seconds: Initial delay between retries
            transient_errors: List of transient error patterns
            permanent_errors: List of permanent error patterns
        """
        self.queue_configs[queue_name] = {
            "max_retries": max_retries,
            "retry_delay": retry_delay_seconds,
            "transient_errors": transient_errors or [
                "timeout",
                "connection",
                "temporarily unavailable",
                "connection pool exhausted"
            ],
            "permanent_errors": permanent_errors or [
                "validation failed",
                "schema mismatch",
                "unknown property",
                "type error"
            ]
        }

        self.metrics["queues"][queue_name] = {
            "total": 0,
            "pending": 0,
            "escalated": 0,
            "resolved": 0
        }

        logger.info(f"Queue configured: {queue_name} "
                   f"(max_retries={max_retries})")

    async def handle_failed_event(
        self,
        event: Dict[str, Any],
        error: Exception,
        queue_name: str,
        agent_id: Optional[str] = None
    ) -> str:
        """
        Handle a failed event by routing to DLQ.

        Args:
            event: Failed event/message
            error: Exception that caused failure
            queue_name: Original queue name
            agent_id: ID of agent that failed

        Returns:
            Message ID for tracking
        """
        # Categorize error
        error_msg = str(error).lower()
        error_category = self._categorize_error(queue_name, error_msg)

        # Create metadata
        metadata = {
            "agent_id": agent_id,
            "original_queue": queue_name,
            "error_category": error_category.value
        }

        # Send to DLQ
        msg_id = await self.dlq_handler.send_to_dlq(
            message=event,
            error=error,
            original_queue=queue_name,
            metadata=metadata,
            error_category=error_category
        )

        # Update metrics
        self.metrics["total_dlq_messages"] += 1
        if queue_name in self.metrics["queues"]:
            self.metrics["queues"][queue_name]["total"] += 1
            self.metrics["queues"][queue_name]["pending"] += 1

        logger.warning(
            f"Event failed and sent to DLQ: {queue_name} "
            f"agent={agent_id} error_type={error_category.value}"
        )

        return msg_id

    async def reprocess_dlq(
        self,
        queue_name: str,
        reprocess_handler: Optional[Any] = None
    ) -> int:
        """
        Reprocess messages in DLQ for a specific queue.

        Args:
            queue_name: Queue name to reprocess
            reprocess_handler: Handler function for reprocessing

        Returns:
            Number of messages successfully processed
        """
        if not reprocess_handler:
            logger.warning(f"No reprocess handler for {queue_name}")
            return 0

        # Reprocess with DLQ handler
        processed = await self.dlq_handler.process_dlq(
            reprocess_handler,
            max_messages=self.config.batch_size
        )

        # Update metrics
        self.metrics["total_processed"] += processed

        logger.info(f"Reprocessed {processed} messages from {queue_name}")
        return processed

    async def get_metrics(self) -> Dict[str, Any]:
        """
        Get current DLQ metrics.

        Returns:
            Metrics dictionary
        """
        stats = await self.dlq_handler.get_dlq_stats()

        self.metrics["total_escalated"] = stats.total_escalated
        self.metrics["current_pending"] = stats.total_pending
        self.metrics["oldest_age_seconds"] = (
            stats.oldest_pending_message_age_seconds
        )

        # Update per-queue metrics
        for queue_name, count in stats.pending_by_queue.items():
            if queue_name in self.metrics["queues"]:
                self.metrics["queues"][queue_name]["pending"] = count

        return self.metrics

    async def get_dlq_status(self, queue_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get DLQ status for queue(s).

        Args:
            queue_name: Specific queue to check (None for all)

        Returns:
            Status dictionary
        """
        stats = await self.dlq_handler.get_dlq_stats()

        if queue_name:
            count = stats.pending_by_queue.get(queue_name, 0)
            return {
                "queue": queue_name,
                "pending": count,
                "status": "healthy" if count < self.config.dlq_depth_threshold else "alert"
            }
        else:
            return {
                "total_pending": stats.total_pending,
                "total_escalated": stats.total_escalated,
                "total_resolved": stats.total_resolved,
                "queues": stats.pending_by_queue
            }

    async def export_prometheus_metrics(self) -> str:
        """
        Export metrics in Prometheus format.

        Returns:
            Prometheus metrics text
        """
        stats = await self.dlq_handler.get_dlq_stats()

        lines = [
            "# HELP greenlang_dlq_depth_gauge Current DLQ depth",
            "# TYPE greenlang_dlq_depth_gauge gauge",
            f"greenlang_dlq_depth_gauge {stats.total_pending}",
            "",
            "# HELP greenlang_dlq_escalated_counter Escalated messages",
            "# TYPE greenlang_dlq_escalated_counter counter",
            f"greenlang_dlq_escalated_counter {stats.total_escalated}",
            "",
            "# HELP greenlang_dlq_resolved_counter Resolved messages",
            "# TYPE greenlang_dlq_resolved_counter counter",
            f"greenlang_dlq_resolved_counter {stats.total_resolved}",
            ""
        ]

        # Per-queue metrics
        for queue, count in stats.pending_by_queue.items():
            safe_queue = queue.replace("-", "_")
            lines.append(
                f"greenlang_dlq_queue_depth_gauge{{queue=\"{queue}\"}} {count}"
            )

        if stats.oldest_pending_message_age_seconds:
            lines.append(
                f"greenlang_dlq_oldest_message_age_seconds "
                f"{stats.oldest_pending_message_age_seconds}"
            )

        return "\n".join(lines)

    # Private methods

    def _categorize_error(self, queue_name: str, error_msg: str) -> ErrorCategory:
        """
        Categorize error as transient or permanent.

        Args:
            queue_name: Queue name
            error_msg: Error message

        Returns:
            Error category
        """
        config = self.queue_configs.get(queue_name, {})

        # Check permanent errors
        for pattern in config.get("permanent_errors", []):
            if pattern.lower() in error_msg:
                return ErrorCategory.PERMANENT

        # Check transient errors
        for pattern in config.get("transient_errors", []):
            if pattern.lower() in error_msg:
                return ErrorCategory.TRANSIENT

        return ErrorCategory.UNKNOWN

    async def _on_threshold_alert(self, stats: DLQStats) -> None:
        """Handle DLQ threshold alert."""
        logger.critical(
            f"DLQ ALERT: Threshold exceeded! "
            f"Pending: {stats.total_pending}, "
            f"Escalated: {stats.total_escalated}"
        )

        # Could send to monitoring system, trigger escalation, etc.
        if self.config.alert_webhook_url:
            await self._send_alert_webhook(stats)

    async def _send_alert_webhook(self, stats: DLQStats) -> None:
        """Send alert to webhook."""
        try:
            import httpx

            payload = {
                "alert": "dlq_threshold_exceeded",
                "pending": stats.total_pending,
                "escalated": stats.total_escalated,
                "timestamp": str(datetime.utcnow())
            }

            async with httpx.AsyncClient() as client:
                await client.post(
                    self.config.alert_webhook_url,
                    json=payload,
                    timeout=5.0
                )

        except Exception as e:
            logger.error(f"Failed to send alert webhook: {e}")

    async def __aenter__(self) -> "ProcessHeatDLQManager":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()


# Example: Integration with agent pipeline
async def example_agent_pipeline_with_dlq():
    """
    Example: Using DLQ handler in a Process Heat agent pipeline.

    Demonstrates how to integrate DLQ handling into the agent workflow.
    """

    # Configure DLQ handler
    config = DLQHandlerConfig(
        kafka_brokers=["localhost:9092"],
        redis_url="redis://localhost:6379",
        max_retries=3,
        initial_backoff_seconds=60,
        dlq_depth_threshold=100
    )

    # Create DLQ manager
    async with ProcessHeatDLQManager(config) as manager:
        # Configure queues
        manager.configure_queue(
            "heat-calculations",
            max_retries=3,
            retry_delay_seconds=60
        )
        manager.configure_queue(
            "emissions-processing",
            max_retries=5,
            retry_delay_seconds=120
        )

        # Example: Handle a failed event
        failed_event = {
            "equipment_id": "boiler_001",
            "temperature": 95.5,
            "pressure": 2.0,
            "timestamp": "2025-12-07T10:30:00Z"
        }

        try:
            # Process event in agent
            result = await process_heat_agent(failed_event)
        except Exception as e:
            # Send to DLQ on failure
            msg_id = await manager.handle_failed_event(
                failed_event,
                e,
                "heat-calculations",
                agent_id="gl_010"
            )
            logger.info(f"Event sent to DLQ: {msg_id}")

        # Later: Reprocess DLQ messages
        async def heat_agent_handler(msg: DLQMessage) -> bool:
            """Handler for reprocessing heat calculation messages."""
            try:
                result = await process_heat_agent(msg.message_body)
                return result is not None
            except Exception:
                return False

        processed = await manager.reprocess_dlq(
            "heat-calculations",
            heat_agent_handler
        )

        # Get metrics
        metrics = await manager.get_metrics()
        print(f"DLQ Metrics: {metrics}")

        # Export Prometheus metrics
        prom_metrics = await manager.export_prometheus_metrics()
        print(f"Prometheus Metrics:\n{prom_metrics}")


async def process_heat_agent(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Simulated Process Heat agent."""
    # Simulate processing
    await asyncio.sleep(0.1)
    return {"status": "processed"}


if __name__ == "__main__":
    # Run example
    asyncio.run(example_agent_pipeline_with_dlq())
